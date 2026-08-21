#pragma once

#include <string>
#include <vector>

#include "rcdl/preproc/geometry.h"

namespace rcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

// ===========================================================================
// PP-OCR: DBNet text detection + CRNN/CTC text recognition
// ===========================================================================
//
// Two INDEPENDENT models, and this header keeps them independent:
//
//   detection   a fully-convolutional DB head emits ONE probability map at the
//               model-input resolution. Threshold it, take the connected
//               regions, fit a minimum-area rectangle to each, expand that
//               rectangle outward by the "unclip" distance, and you have the
//               text quadrilaterals.
//   recognition a CRNN-style sequence model emits [T, C] per-step class scores
//               for one already-cropped, already-deskewed text LINE. Collapse
//               the best path with the CTC rule and look the surviving indices
//               up in a character dictionary.
//   orientation a two-class head says whether that line crop is upright or
//               upside down, which the recogniser cannot tell you: a CTC model
//               fed a 180-degree-rotated line does not fail, it reads out
//               confident nonsense. See TextAngleClassifier.
//
// The middle step — crop each detected quadrilateral out of the ORIGINAL frame,
// warp it upright, and letterbox it into the recogniser's input — is a host
// image operation, not a decode, so it is deliberately NOT assembled here. The
// application (or the Python layer) composes the two stages; see sortTextBoxes()
// for the reading order the reference pipeline feeds the recogniser in.
//
// Everything below is Engine-free and OpenCV-free apart from the two thin
// Engine-bound wrappers at the bottom: the decoders take plain float buffers, so
// the numpy tests in tests/test_ocr.py are the reference implementation and the
// C++ is checked against them.

/// One detected text region as a QUADRILATERAL, in ORIGINAL-image pixel
/// coordinates (already un-letterboxed back from the model-input canvas).
///
/// A quadrilateral rather than a box because text lines are rarely axis-aligned:
/// the DB head's regions are fitted with a ROTATED minimum-area rectangle, and
/// flattening that to its extent would hand the recogniser a crop full of
/// neighbouring lines. `pts` is that rectangle's four corners in the reference
/// decoder's order — top-left, top-right, bottom-right, bottom-left — which is
/// also the order a perspective crop expects, so it draws as a closed polyline
/// and warps upright without re-ordering.
///
/// `x1..y2` is the axis-aligned bounding box of `pts`, kept alongside because
/// filtering and coarse cropping want it and recomputing it per use is noise.
struct TextBox {
  float pts[8];  ///< x0,y0, x1,y1, x2,y2, x3,y3 — TL, TR, BR, BL
  float x1;      ///< axis-aligned bbox left   (min x of pts)
  float y1;      ///< axis-aligned bbox top    (min y of pts)
  float x2;      ///< axis-aligned bbox right  (max x of pts)
  float y2;      ///< axis-aligned bbox bottom (max y of pts)
  float score;   ///< mean probability-map value inside the region, in [0,1]
};

// ---------------------------------------------------------------------------
// Quadrilateral geometry — the two pieces the DB post-process is built from
// ---------------------------------------------------------------------------

/// Fit the minimum-area enclosing rectangle of a point set and write its four
/// corners to `out` (TL, TR, BR, BL). `short_side`, when non-null, receives the
/// rectangle's shorter side length — the reference filters regions on it.
///
/// `xy` is `n` interleaved x,y pairs. Convex hull (monotone chain) followed by
/// rotating calipers: the minimum-area enclosing rectangle of a convex polygon
/// always has one side FLUSH with a hull edge, so trying every hull edge as the
/// rectangle's axis and keeping the smallest area is exact, not a search. Cost
/// is O(h²) in the number of HULL vertices, which for a text blob is a few tens.
///
/// Corner ordering is the reference's: sort the four corners by x, then within
/// the left pair the upper one is TL and within the right pair the upper one is
/// TR. Degenerate input (n < 1) yields four zero corners.
void minAreaQuad(const float* xy, int n, float out[8], float* short_side = nullptr);

/// Expand a convex quadrilateral outward by the DB "unclip" distance
///
///     d = area * ratio / perimeter
///
/// which is the offset PaddleOCR applies with a polygon clipper. The DB head is
/// trained to predict a SHRUNK version of each text region (that is what makes
/// adjacent lines separable in the probability map), so post-processing has to
/// grow it back or every crop clips its own characters.
///
/// Implemented as a miter offset: each corner moves along the bisector of its
/// two edge normals by exactly the distance that pushes BOTH adjacent edges out
/// by `d`. For the rectangle this is always called with, that is provably the
/// same answer as the clipper's — offsetting a w x h rectangle by d and taking
/// the minimum-area rectangle of the result gives (w + 2d) x (h + 2d) whatever
/// the join style — so no clipper dependency is needed to match the reference.
///
/// `in` and `out` are 4 corners as x0,y0,..,x3,y3 and may alias.
void unclipQuad(const float in[8], float ratio, float out[8]);

// ---------------------------------------------------------------------------
// A. Detection (DBNet)
// ---------------------------------------------------------------------------

/// DB probability-map post-processing parameters. The defaults are the
/// reference PP-OCR detection configuration.
struct OcrDetConfig {
  /// Binarisation threshold on the [0,1] probability map: foreground is
  /// prob > bin_thresh. Controls how far a region GROWS before unclipping.
  float bin_thresh = 0.3f;
  /// Minimum mean probability inside a fitted rectangle for it to be kept.
  /// Computed over the rectangle's own mask, not its extent, so a diagonal line
  /// is not punished for the empty corners of its bounding box.
  float box_thresh = 0.6f;
  /// Unclip expansion ratio; larger => fatter boxes. 1.5 for detection feeding a
  /// recogniser, ~1.8-2.0 when the crops come out visibly clipped.
  float unclip_ratio = 1.5f;
  /// Drop a region whose fitted rectangle's shorter side is below this many
  /// probability-map pixels (the reference's min_size; the check is repeated
  /// after unclipping against min_size + 2).
  int min_size = 3;
  /// Drop a final box whose width or height in ORIGINAL-image pixels is at most
  /// this — the reference's last filter, which removes slivers that only became
  /// slivers once mapped back through a large downscale.
  int min_box_side = 3;
  /// Cap on how many regions are considered, highest-first is NOT applied — the
  /// reference simply takes the first N contours it finds, and so do we, so that
  /// a pathological map cannot turn one frame into a minute of hull fitting.
  int max_candidates = 1000;
  /// Connectivity used to group foreground pixels into regions: 8 (the
  /// reference's contour connectivity) or 4.
  int connectivity = 8;
  /// Apply a sigmoid to the map before thresholding.
  ///
  /// FALSE by default because every PP-OCR detection export ends WITH the
  /// sigmoid in the graph (the output tensor is literally the Sigmoid node, and
  /// on a quantized export it is scaled 1/255 over [0,1] — TextDetector prints
  /// that in describe()). Set it only for a head exported to raw logits. How to
  /// tell them apart: dump the map's min/max — a probability map sits inside
  /// [0,1] and piles up at both ends, logits run negative and above 1.
  bool apply_sigmoid = false;
};

/// Decode a DB probability map into text quadrilaterals.
///
/// `prob` : row-major [H, W] floats, the map with its unit dims dropped.
///          prob[y*W + x]. In [0,1] unless cfg.apply_sigmoid is set.
/// `H,W`  : map dimensions. NOT taken from a shape here on purpose — resolving
///          [1,1,H,W] vs [1,H,W,1] needs the tensor's format, which only the
///          Engine-bound wrapper has; see TextDetector.
/// `lb`   : the letterbox geometry used at preprocess time. Map pixels are first
///          scaled to model-input pixels by (lb.dstW/W, lb.dstH/H) — a DB head
///          may emit at a lower resolution than its input — then un-letterboxed
///          through lb.invX/invY and clamped to the source extent. A
///          default-constructed `lb` (dstW/dstH == 0) means "identity", i.e.
///          coordinates come back in probability-map pixels.
///
/// Pipeline, mirroring the reference decoder step for step:
///   binarise -> connected regions -> minimum-area rectangle per region ->
///   drop short_side < min_size -> mean-probability score + box_thresh ->
///   unclip -> drop short_side < min_size+2 -> map to original pixels ->
///   drop degenerate sides.
///
/// One deliberate difference from the reference: regions are found by connected
/// components rather than by contour tracing, so a ring-shaped region (a text
/// box drawn as an outline) yields ONE region here and two contours there. For
/// text that is a difference without a distinction, and it avoids inventing a
/// second box inside the first.
std::vector<TextBox> decodeTextBoxes(const float* prob, int H, int W,
                                     const OcrDetConfig& cfg, const LetterboxInfo& lb);

/// Sort boxes into reading order: top to bottom, then left to right.
///
/// A plain lexicographic sort on (y, x) would scramble a line whose boxes differ
/// by a pixel or two of top edge, so — as in the reference — boxes are first
/// sorted by (top-left y, top-left x) and then a single insertion pass swaps
/// neighbours that belong to the same visual ROW (their top edges within
/// `row_tol` pixels) but are out of left-to-right order.
void sortTextBoxes(std::vector<TextBox>& boxes, float row_tol = 10.0f);

/// Engine-bound DB text detector.
///
/// The caller preprocesses + infer()s; postprocess() reads the selected output
/// through outputAsFloat() (dequantizing the usual int8-affine map) and runs
/// decodeTextBoxes(). The map's H/W come from the output's own
/// rknn_tensor_attr, never from `cfg`.
class TextDetector {
 public:
  explicit TextDetector(Engine& engine, OcrDetConfig cfg = OcrDetConfig(),
                        int output_index = 0);

  std::vector<TextBox> postprocess(const LetterboxInfo& lb) const;

  const OcrDetConfig& config() const { return cfg_; }
  int mapWidth() const { return map_w_; }
  int mapHeight() const { return map_h_; }
  /// "480x480 map, i8 affine scale=0.003922 zp=-128" — what the constructor
  /// resolved, for a demo to print and for a bug report to quote.
  std::string describe() const;

 private:
  Engine& engine_;
  OcrDetConfig cfg_;
  int out_idx_;
  int map_h_ = 0;
  int map_w_ = 0;
};

// ---------------------------------------------------------------------------
// B. Recognition (CRNN + CTC greedy decode)
// ---------------------------------------------------------------------------

/// Load a character dictionary, one token per line, UTF-8 bytes kept verbatim
/// (a trailing CR from a CRLF file is stripped). `dict[i]` is the token for
/// class index `i`.
///
/// INDEX CONVENTION: the CTC blank lives at class 0 and an emitted class `i` is
/// looked up as `dict[i]` directly — NOT `dict[i-1]`. A raw PaddleOCR
/// `ppocr_keys_*.txt` does NOT satisfy that: it is the bare character list, and
/// the reference decoder prepends a blank token and appends a space token to it
/// at load time. Pass `paddle_special = true` to apply exactly that, or ship a
/// dictionary that already carries both (which is what the files in `data/` are
/// — their line count is the model's class count, blank and space included).
///
/// Throws rcdl::Error when the file cannot be opened, because a silently empty
/// dictionary decodes every image to the empty string.
std::vector<std::string> loadCharDict(const std::string& path, bool paddle_special = false);

/// One recognised text line.
struct TextLine {
  std::string text;  ///< decoded UTF-8, possibly empty
  float score;       ///< mean per-step confidence over the steps that emitted a
                     ///< character, 0 when nothing was emitted. In [0,1] when
                     ///< the head emits probabilities (see OcrRecConfig).
};

/// Recognition head parameters.
struct OcrRecConfig {
  /// Sequence-axis order of the output tensor: true => [1, T, C] (T outer),
  /// false => [1, C, T]. TextRecognizer OVERRIDES this when one of the two axes
  /// matches the dictionary size exactly — see its constructor.
  bool time_major = true;
  /// Softmax each step before reading its confidence.
  ///
  /// FALSE by default because the PP-OCR recognition exports end WITH the
  /// softmax in the graph, and because the softmax cannot change the argmax —
  /// it only changes what `score` MEANS. Leave it false and a logit-emitting
  /// head still decodes the right text, but hands back an unbounded score. How
  /// to tell them apart: one step's values sum to 1 and are all non-negative iff
  /// the softmax is already in the graph.
  bool apply_softmax = false;
  /// Class index of the CTC blank. 0 for every PaddleOCR dictionary.
  int blank_index = 0;
};

/// CTC best-path (greedy) decode of a recognition head's per-step scores.
///
/// `logits`      : row-major [num_steps, num_classes], i.e. logits[t*C + c].
/// `num_steps`   : T, the sequence length.
/// `num_classes` : C, the vocabulary INCLUDING the blank.
/// `dict`        : token table indexed by class id (blank at cfg.blank_index).
///
/// THE COLLAPSE RULE, which is the whole of CTC decoding: walk the per-step
/// argmax and emit a class when it is (a) not the blank and (b) not equal to the
/// PREVIOUS step's argmax — blank or not. So `a a` emits one "a" and `a blank a`
/// emits two: the blank is not a character, it is the SEPARATOR that lets a
/// doubled letter survive. Dropping blanks first and de-duplicating afterwards
/// would turn "aa" into "a" and is the classic way to get this wrong.
///
/// `score` is the mean of the per-step winning value over exactly the steps that
/// emitted a character. An out-of-range class id (a dictionary shorter than the
/// head's class count) is skipped rather than read out of bounds.
TextLine ctcGreedyDecode(const float* logits, int num_steps, int num_classes,
                         const std::vector<std::string>& dict,
                         const OcrRecConfig& cfg = OcrRecConfig());

/// Engine-bound text recogniser for ONE already-cropped, upright text line.
///
/// The constructor loads the dictionary and resolves the output's sequence and
/// class axes from the model (see below); postprocess() reads the output through
/// outputAsFloat() and runs ctcGreedyDecode().
///
/// HOW THE AXES ARE TOLD APART. A rec output is [1, T, C] or [1, C, T] and the
/// tensor format does not say which — a 3-D tensor is reported UNDEFINED. What
/// does say is the DICTIONARY: C is the model's class count, so the axis whose
/// length equals dict().size() is the class axis, and the other is time. That is
/// checked first and wins. When neither axis matches (a mismatched dictionary,
/// which is worth noticing rather than papering over) the fallback is
/// cfg.time_major, whose default matches every PP-OCR export.
class TextRecognizer {
 public:
  explicit TextRecognizer(Engine& engine, const std::string& dict_path,
                          OcrRecConfig cfg = OcrRecConfig(), int output_index = 0);
  /// Overload taking an already-loaded dictionary — the right one when several
  /// Engines (one per NPU core) share a 18k-entry table.
  TextRecognizer(Engine& engine, std::vector<std::string> dict,
                 OcrRecConfig cfg = OcrRecConfig(), int output_index = 0);

  TextLine postprocess() const;

  const std::vector<std::string>& dict() const { return dict_; }
  const OcrRecConfig& config() const { return cfg_; }
  int numSteps() const { return num_steps_; }
  int numClasses() const { return num_classes_; }
  /// True when the output is [1, T, C]; false when it is [1, C, T].
  bool timeMajor() const { return cfg_.time_major; }
  /// "T=40 C=6625 [1,T,C], dict 6625" — the resolution the constructor made.
  std::string describe() const;

 private:
  void resolveAxes();

  Engine& engine_;
  std::vector<std::string> dict_;
  OcrRecConfig cfg_;
  int out_idx_;
  int num_steps_ = 0;
  int num_classes_ = 0;
};

// ===========================================================================
// Text-line orientation (0 deg / 180 deg)
// ===========================================================================
//
// WHY THIS EXISTS, since it is easy to leave out: nothing else in the pipeline
// notices an upside-down line. The detector finds the region either way — a
// minimum-area rectangle has no top — and the recogniser reads whatever it is
// given, confidently. A page photographed upside down, a rotated ID card, a
// label read from the wrong side: without this head they decode to plausible,
// wrong strings, with a good score, silently. That is why PP-OCR's own pipeline
// puts a direction classifier between cropping and recognition.
//
// The head is a two-class classifier over the SAME crop the recogniser gets,
// with classes {0 deg, 180 deg}. It cannot detect 90 deg: those come out of the
// detector as tall quadrilaterals and are handled by the crop-and-warp step,
// which rotates them upright — leaving exactly the 180 deg ambiguity this
// resolves.

/// How wide a line crop lands in a WxH model input under PP-OCR's own fit.
///
/// The rule, and it is not the letterbox the rest of the library uses: scale so
/// the crop fills the input's HEIGHT, cap the width at the input's, anchor the
/// result at the TOP-LEFT and leave the remainder as padding. Only a line
/// narrower than the input's aspect ratio keeps its shape; a wide one — which a
/// text line almost always is — is squashed to the full width.
///
/// It matters more than a preprocessing detail usually does. Feeding this head a
/// CENTRED letterbox instead, measured on the 16 lines of the sample page,
/// drops it from 16/16 orientations right to 9/16 upright and 11/16 rotated,
/// with mean confidence 0.98 -> 0.78 — the model is looking at a thin strip of
/// text between two thick bars, which is not what it was trained on. Nothing
/// errors; the answers just get worse.
///
/// Returns the destination width in [1, dst_w]; the caller fills
/// [width, dst_w) with the pad value.
int ocrLineFitWidth(int src_w, int src_h, int dst_w, int dst_h);

/// A text line's orientation verdict.
struct TextOrientation {
  int label = 0;         ///< 0 = upright, 1 = rotated by 180 degrees
  float score = 0.0f;    ///< the winning class's value (a probability when the
                         ///< export ends in softmax, which PP-OCR's does)
  bool flip180 = false;  ///< label == 1 AND score > threshold: rotate the crop
                         ///< before handing it to the recogniser
};

/// Decode a direction head's output into a verdict.
///
/// `scores` is `n` values ([1,2] with the batch dim dropped). The argmax is the
/// label and its value the score; `flip180` is set when the label is 1 AND the
/// score exceeds `thresh`.
///
/// THE THRESHOLD IS ASYMMETRIC ON PURPOSE, and it is the whole design of this
/// head's use: flipping an upright line makes it unreadable, while leaving a
/// rotated one alone is no worse than not having the classifier. So the gate
/// applies to the flip decision only — PP-OCR's default of 0.9 means "flip when
/// nearly certain", and `label` still reports what the model actually said.
/// Returns {0, 0, false} for an empty buffer.
TextOrientation decodeTextOrientation(const float* scores, int n, float thresh = 0.9f);

/// Engine-bound 0/180-degree classifier for ONE already-cropped text line.
///
/// Postprocess-only, like TextRecognizer: the caller (or the Python layer) does
/// the crop and the resize into the model's input, because a crop is a host
/// image operation. Reads output[output_index] through outputAsFloat(), so an
/// int8 export is dequantized like any other.
class TextAngleClassifier {
 public:
  explicit TextAngleClassifier(Engine& engine, float thresh = 0.9f, int output_index = 0);

  TextOrientation postprocess() const;

  float threshold() const noexcept { return thresh_; }
  /// "2 classes, flip above 0.90" — what the constructor resolved.
  std::string describe() const;

 private:
  Engine& engine_;
  float thresh_;
  int out_idx_;
};

}  // namespace rcdl
