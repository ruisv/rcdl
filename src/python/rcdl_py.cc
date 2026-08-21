// nanobind bindings for RCDL.
//
// The data path stays explicit: inputs come in as C-contiguous numpy arrays
// (raw bytes copied into the NPU input buffer), outputs go out as float32
// numpy arrays (dequantized) or raw bytes + dtype. The pure-python `rcdl`
// wrapper (python/rcdl/__init__.py) adds the numpy conveniences on top.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "rcdl/backend/engine.h"
#include "rcdl/backend/output_reader.h"
#include "rcdl/core/dma_buf.h"
#include "rcdl/media/jpeg_codec.h"
#include "rcdl/media/video_codec.h"
#include "rcdl/media/video_frame.h"
#include "rcdl/pipeline/async_video_detection_pipeline.h"
#include "rcdl/pipeline/detection_pipeline.h"
#include "rcdl/pipeline/tracking_pipeline.h"
#include "rcdl/preproc/image.h"
#include "rcdl/preproc/letterbox.h"
#include "rcdl/preproc/rga.h"
#include "rcdl/tasks/classification.h"
#include "rcdl/tasks/depth.h"
#include "rcdl/tasks/detection.h"
#include "rcdl/tasks/embedding.h"
#include "rcdl/tasks/face.h"
#include "rcdl/tasks/instance_seg.h"
#include "rcdl/tasks/obb.h"
#include "rcdl/tasks/ocr.h"
#include "rcdl/tasks/pose.h"
#include "rcdl/tasks/segmentation.h"
#include "rcdl/tracks/byte_tracker.h"
#include "rcdl/tracks/reid.h"
#include "rcdl/version.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using Contig = nb::ndarray<nb::numpy, nb::c_contig, nb::device::cpu>;

void setInputFromArray(rcdl::Engine& e, int i, const Contig& arr) {
  std::size_t bytes = arr.itemsize();
  for (std::size_t d = 0; d < arr.ndim(); ++d) bytes *= arr.shape(d);
  e.setInput(i, arr.data(), bytes);
}

nb::ndarray<nb::numpy, float> outputFloat(const rcdl::Engine& e, int i) {
  const auto& a = e.outputAttr(i);
  std::vector<std::size_t> shape(a.dims, a.dims + a.n_dims);
  if (shape.empty()) shape.push_back(a.n_elems);
  float* buf = new float[a.n_elems];
  e.outputAsFloat(i, buf, a.n_elems);
  nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<float*>(p); });
  return nb::ndarray<nb::numpy, float>(buf, shape.size(), shape.data(), owner);
}


// --- M1: preproc + detection helpers -----------------------------------------

using LbTuple = std::tuple<float, float, float, int, int, int, int>;

rcdl::LetterboxInfo lbFromTuple(const LbTuple& t) {
  rcdl::LetterboxInfo lb;
  lb.scale = std::get<0>(t);
  lb.padX = std::get<1>(t);
  lb.padY = std::get<2>(t);
  lb.srcW = std::get<3>(t);
  lb.srcH = std::get<4>(t);
  lb.dstW = std::get<5>(t);
  lb.dstH = std::get<6>(t);
  return lb;
}

LbTuple lbToTuple(const rcdl::LetterboxInfo& lb) {
  return LbTuple(lb.scale, lb.padX, lb.padY, lb.srcW, lb.srcH, lb.dstW, lb.dstH);
}

rcdl::PixelFormat formatFromName(const std::string& n) {
  if (n == "rgb888") return rcdl::PixelFormat::RGB888;
  if (n == "bgr888") return rcdl::PixelFormat::BGR888;
  if (n == "rgba8888") return rcdl::PixelFormat::RGBA8888;
  if (n == "bgra8888") return rcdl::PixelFormat::BGRA8888;
  if (n == "gray8") return rcdl::PixelFormat::GRAY8;
  if (n == "nv12") return rcdl::PixelFormat::NV12;
  if (n == "nv21") return rcdl::PixelFormat::NV21;
  if (n == "yuv420p") return rcdl::PixelFormat::YUV420P;
  throw std::invalid_argument("unknown pixel format: " + n);
}

// The Python API names formats and codecs with the lower-case tokens
// formatFromName()/codecFromName() accept, so a value read off an object can be
// passed straight back in (`rcdl.letterbox(x, ..., src_fmt=frame.format)`).
// The C++ formatName()/codecName() spellings are for human-readable messages.
const char* pyFormatName(rcdl::PixelFormat f) noexcept {
  switch (f) {
    case rcdl::PixelFormat::RGB888: return "rgb888";
    case rcdl::PixelFormat::BGR888: return "bgr888";
    case rcdl::PixelFormat::RGBA8888: return "rgba8888";
    case rcdl::PixelFormat::BGRA8888: return "bgra8888";
    case rcdl::PixelFormat::GRAY8: return "gray8";
    case rcdl::PixelFormat::NV12: return "nv12";
    case rcdl::PixelFormat::NV21: return "nv21";
    case rcdl::PixelFormat::YUV420P: return "yuv420p";
    case rcdl::PixelFormat::Unknown: break;
  }
  return "unknown";
}

const char* pyCodecName(rcdl::VideoCodec c) noexcept {
  switch (c) {
    case rcdl::VideoCodec::H264: return "h264";
    case rcdl::VideoCodec::H265: return "h265";
    case rcdl::VideoCodec::VP8: return "vp8";
    case rcdl::VideoCodec::VP9: return "vp9";
    case rcdl::VideoCodec::AV1: return "av1";
    case rcdl::VideoCodec::MJPEG: return "mjpeg";
  }
  return "unknown";
}

rcdl::PreprocBackend backendFromName(const std::string& n) {
  if (n == "auto") return rcdl::PreprocBackend::Auto;
  if (n == "rga") return rcdl::PreprocBackend::Rga;
  if (n == "cpu") return rcdl::PreprocBackend::Cpu;
  throw std::invalid_argument("unknown preproc backend: " + n);
}

// `Contig` is dtype-AGNOSTIC (nanobind only constrains contiguity and device),
// so every raw reinterpret below has to check the dtype itself. Without this a
// float64 array — numpy's default for a Python list of numbers — is silently
// read as float32 and produces nonsense, and a float16 array reads twice past
// the end of the buffer.
const float* floatData(const Contig& a, const char* what) {
  if (a.dtype() != nb::dtype<float>()) {
    throw std::invalid_argument(std::string(what) + ": expected a float32 array");
  }
  return static_cast<const float*>(a.data());
}

std::size_t elemCount(const Contig& a) {
  std::size_t n = 1;
  for (std::size_t d = 0; d < a.ndim(); ++d) n *= a.shape(d);
  return n;
}

// A uint8 numpy array viewed as an image. Planar YUV arrives flat (or as an
// (H*3/2, W) array); packed formats arrive as (H, W, C) — either way the array
// is only a byte container, so the caller states width/height/format.
rcdl::ImageView viewFromArray(const Contig& a, int width, int height,
                              const std::string& fmt, int wstride, int hstride) {
  const rcdl::PixelFormat f = formatFromName(fmt);
  if (a.dtype() != nb::dtype<std::uint8_t>()) {
    throw std::invalid_argument("image arrays must be uint8");
  }
  rcdl::ImageView v = rcdl::hostView(const_cast<void*>(a.data()), width, height, f,
                                     wstride, hstride);
  std::size_t bytes = a.itemsize();
  for (std::size_t d = 0; d < a.ndim(); ++d) bytes *= a.shape(d);
  if (bytes < v.bytes()) {
    throw std::invalid_argument("array holds " + std::to_string(bytes) + " bytes but " +
                                v.describe() + " needs " + std::to_string(v.bytes()));
  }
  v.size = bytes;
  return v;
}

rcdl::VideoCodec codecFromName(const std::string& n) {
  if (n == "h264") return rcdl::VideoCodec::H264;
  if (n == "h265" || n == "hevc") return rcdl::VideoCodec::H265;
  if (n == "vp8") return rcdl::VideoCodec::VP8;
  if (n == "vp9") return rcdl::VideoCodec::VP9;
  if (n == "av1") return rcdl::VideoCodec::AV1;
  if (n == "mjpeg" || n == "jpeg") return rcdl::VideoCodec::MJPEG;
  throw std::invalid_argument("unknown video codec: " + n);
}

rcdl::RcMode rcFromName(const std::string& n) {
  if (n == "cbr") return rcdl::RcMode::Cbr;
  if (n == "vbr") return rcdl::RcMode::Vbr;
  if (n == "fixqp") return rcdl::RcMode::FixQp;
  if (n == "avbr") return rcdl::RcMode::AvBr;
  throw std::invalid_argument("unknown rate-control mode: " + n);
}

// receive() shapes: None when nothing is ready, so a caller can poll in a loop.
template <typename Dec>
std::unique_ptr<rcdl::VideoFrame> receiveFrame(Dec& d, int timeout_ms) {
  auto f = std::make_unique<rcdl::VideoFrame>();
  bool got;
  {
    nb::gil_scoped_release nogil;
    got = d.receive(*f, timeout_ms);
  }
  return got ? std::move(f) : nullptr;
}

template <typename Enc>
nb::object receivePacket(Enc& e, int timeout_ms) {
  std::vector<std::uint8_t> out;
  bool got;
  {
    nb::gil_scoped_release nogil;
    got = e.receive(out, timeout_ms);
  }
  if (!got) return nb::none();
  return nb::bytes(reinterpret_cast<const char*>(out.data()), out.size());
}

// Every Engine-bound head takes "an Engine", and Python spells that two ways:
// the compiled `rcdl_py.Engine`, and the `rcdl.Engine` wrapper that holds one in
// `_e`. Accept both. The alternative is that `rcdl.Segmenter(engine)` — the
// obvious thing to write once `rcdl.Engine` exists — fails with an
// argument-type error mentioning neither the wrapper nor `_e`.
//
// Lifetime: every constructor below also carries nb::keep_alive<1, 2>, which
// pins whichever OBJECT was passed. When that is the wrapper, the wrapper owns
// `_e`, so the C++ Engine this reference points at outlives the head either way.
rcdl::Engine& engineFrom(nb::handle engine) {
  if (nb::hasattr(engine, "_e")) return nb::cast<rcdl::Engine&>(engine.attr("_e"));
  return nb::cast<rcdl::Engine&>(engine);
}

// --- M4: task-head helpers ----------------------------------------------------

// An array's shape as the decoders want it. The task decoders take the tensor
// shape as a vector<int> and resolve the layout themselves, so passing the
// numpy shape through means a [1,1000] and a [1,1,1,1000] classifier head are
// both understood — and a tensor that is not a score vector at all is rejected
// there instead of being flattened into nonsense.
std::vector<int> shapeOf(const Contig& a) {
  std::vector<int> s;
  s.reserve(a.ndim());
  for (std::size_t d = 0; d < a.ndim(); ++d) s.push_back(static_cast<int>(a.shape(d)));
  return s;
}

const std::int32_t* int32Data(const Contig& a, const char* what) {
  if (a.dtype() != nb::dtype<std::int32_t>()) {
    throw std::invalid_argument(std::string(what) + ": expected an int32 array");
  }
  return static_cast<const std::int32_t*>(a.data());
}

// Masks, label maps and depth maps go out as numpy arrays of the map's own
// shape, never as flat buffers: a (h*w,) vector plus a width the caller has to
// remember is exactly how a mask ends up transposed. Always a COPY — the C++
// object stays the owner of its storage and may be a temporary.
template <typename T>
nb::ndarray<nb::numpy, T> ownedArray(const T* src, std::vector<std::size_t> shape) {
  std::size_t n = 1;
  for (std::size_t s : shape) n *= s;
  T* buf = new T[n > 0 ? n : 1];
  if (n > 0 && src != nullptr) std::memcpy(buf, src, n * sizeof(T));
  nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<T*>(p); });
  return nb::ndarray<nb::numpy, T>(buf, shape.size(), shape.data(), owner);
}

// Shape a decoder's output vector into (h, w[, c]), falling back to an empty
// array when the vector does not hold the whole map (masks are optional, and
// `compute_masks=False` leaves the geometry set but the buffer empty).
template <typename T>
nb::ndarray<nb::numpy, T> shapedArray(const std::vector<T>& v, std::vector<std::size_t> shape) {
  std::size_t n = 1;
  for (std::size_t s : shape) n *= s;
  if (n != v.size()) return ownedArray<T>(nullptr, {0, 0});
  return ownedArray<T>(v.data(), std::move(shape));
}

rcdl::SegScore segScoreFromName(const std::string& n) {
  if (n == "none") return rcdl::SegScore::kNone;
  if (n == "softmax") return rcdl::SegScore::kSoftmax;
  if (n == "max") return rcdl::SegScore::kMax;
  throw std::invalid_argument("unknown segmentation score mode: " + n);
}

// A 2-D int32 label array as a SegMask, so seg_colorize()/seg_resize() work on
// a plain numpy map and not only on what decode_seg() returned.
rcdl::SegMask segMaskFromArray(const Contig& labels) {
  if (labels.ndim() != 2) {
    throw std::invalid_argument("expected an (h, w) int32 label array");
  }
  const std::int32_t* p = int32Data(labels, "label map");
  rcdl::SegMask m;
  m.height = static_cast<int>(labels.shape(0));
  m.width = static_cast<int>(labels.shape(1));
  m.labels.assign(p, p + elemCount(labels));
  std::int32_t hi = -1;
  for (std::int32_t v : m.labels) hi = std::max(hi, v);
  m.num_classes = hi + 1;
  return m;
}

// The same for a float32 depth map.
rcdl::DepthMap depthMapFromArray(const Contig& data) {
  if (data.ndim() != 2) {
    throw std::invalid_argument("expected an (h, w) float32 depth array");
  }
  const float* p = floatData(data, "depth map");
  rcdl::DepthMap m;
  m.height = static_cast<int>(data.shape(0));
  m.width = static_cast<int>(data.shape(1));
  m.data.assign(p, p + elemCount(data));
  if (!m.data.empty()) {
    m.vmin = *std::min_element(m.data.begin(), m.data.end());
    m.vmax = *std::max_element(m.data.begin(), m.data.end());
  }
  return m;
}

// --- M4b: pose / OBB / OCR / face helpers -------------------------------------

rcdl::KeypointDecode kptDecodeFromName(const std::string& n) {
  if (n == "model_pixels") return rcdl::KeypointDecode::kModelPixels;
  if (n == "cell_relative") return rcdl::KeypointDecode::kCellRelative;
  throw std::invalid_argument("unknown keypoint decode: " + n);
}

const char* pyKptDecodeName(rcdl::KeypointDecode d) noexcept {
  return d == rcdl::KeypointDecode::kCellRelative ? "cell_relative" : "model_pixels";
}

// A rotated box as the 5-tuple Python spells it, so `rotated_iou((cx, cy, w, h,
// a), ...)` works without constructing a RotatedBox first — the same courtesy
// nms() extends by taking an (N, 6) array instead of a list of Detections.
using RrTuple = std::tuple<float, float, float, float, float>;

rcdl::RotatedBox rrFromTuple(const RrTuple& t) {
  rcdl::RotatedBox r;
  r.cx = std::get<0>(t);
  r.cy = std::get<1>(t);
  r.w = std::get<2>(t);
  r.h = std::get<3>(t);
  r.angle = std::get<4>(t);
  return r;
}

// The per-scale buffers of a multi-branch head, checked the way
// decode_yolo_ltrb checks them: the decoder indexes with the CALLER's grid,
// class and reg_max numbers, and nothing in numpy ties those to the arrays'
// real sizes, so an inconsistent call would read far out of bounds.
void requireSameLength(std::size_t n, std::size_t want, const char* what) {
  if (n != want) throw std::invalid_argument(what);
}

void requireElems(const Contig& a, std::size_t need, const char* what, std::size_t scale) {
  if (elemCount(a) < need) {
    throw std::invalid_argument(std::string(what) + ": scale " + std::to_string(scale) +
                                " needs " + std::to_string(need) + " elements, got " +
                                std::to_string(elemCount(a)));
  }
}

// A probability / logit map as (H, W). Unit axes are dropped first, so the
// [1,1,H,W] of an NCHW export and the [1,H,W,1] of an NHWC one both resolve —
// the same reconciliation TextDetector does from the tensor's own attr.
std::pair<int, int> mapHW(const Contig& a, const char* what) {
  std::vector<int> dims;
  for (std::size_t d = 0; d < a.ndim(); ++d) {
    const int v = static_cast<int>(a.shape(d));
    if (v != 1) dims.push_back(v);
  }
  if (dims.size() != 2) {
    throw std::invalid_argument(std::string(what) +
                                ": expected an (h, w) map (unit axes are ignored)");
  }
  return {dims[0], dims[1]};
}

// 5 landmarks as (5, 2) or a flat (10,) float32 array — the order the detector
// produces: left eye, right eye, nose, left mouth corner, right mouth corner.
void quadOrFive(const Contig& a, float out[10], const char* what) {
  if (elemCount(a) != 10) {
    throw std::invalid_argument(std::string(what) +
                                ": expected 5 points — a (5, 2) or (10,) float32 array");
  }
  const float* p = floatData(a, what);
  std::copy(p, p + 10, out);
}

// 4 corners as (4, 2) or a flat (8,) float32 array — TL, TR, BR, BL.
void quadFromArray(const Contig& a, float out[8], const char* what) {
  if (elemCount(a) != 8) {
    throw std::invalid_argument(std::string(what) +
                                ": expected 4 corners — a (4, 2) or (8,) float32 array");
  }
  const float* p = floatData(a, what);
  std::copy(p, p + 8, out);
}

// Prior boxes go out as an (N, 4) array of normalized (cx, cy, w, h) rather than
// N little objects: the whole point of a prior set is that it is compared,
// counted and plotted as a block, and 16800 bound instances is not that.
nb::ndarray<nb::numpy, float> priorsArray(const std::vector<rcdl::PriorBox>& priors) {
  std::vector<float> flat;
  flat.reserve(priors.size() * 4);
  for (const rcdl::PriorBox& p : priors) {
    flat.push_back(p.cx);
    flat.push_back(p.cy);
    flat.push_back(p.w);
    flat.push_back(p.h);
  }
  return ownedArray<float>(flat.data(), {priors.size(), 4});
}

/// The Engine-bound text recogniser, which cannot be a BoundTask.
///
/// The deployed PP-OCR recognition export is a FLOAT model — [1,48,320,3] f32
/// in, [1,40,6625] f16 out — so there is no uint8 input tensor for RGA to write
/// into and the zero-copy chain the other heads use does not apply. This is the
/// host path instead: resize the line crop into a packed uint8 scratch buffer
/// (RGA imports the mapping when it will take it, CPU otherwise), convert that
/// into the encoding the model's input actually takes, and hand it over with
/// Engine::setInput. A quantized rec export takes the bytes unchanged; a float
/// one gets `pixel * scale + shift`, because the mean/std a quantized export
/// carries inside its graph is not there to be folded into.
///
/// It also RESIZES rather than letterboxes by default: a CRNN line crop is
/// already the region of interest, every PP-OCR recognition export is fed a
/// plain resize to the model's own WxH, and padding a line out to the input's
/// aspect ratio would spend most of the sequence axis on background.
struct BoundRecognizer {
  rcdl::Engine* engine;
  int in_w = 0;
  int in_h = 0;
  rcdl::PixelFormat in_fmt;
  bool float_input = false;
  float scale = 1.0f / 255.0f;
  float shift = 0.0f;
  bool stretch = true;
  std::uint8_t pad = 0;
  rcdl::PreprocBackend backend;
  rcdl::PreprocBackend last_backend = rcdl::PreprocBackend::Auto;
  std::vector<std::uint8_t> host;   ///< packed WxHxC scratch, reused per call
  std::vector<float> floats;        ///< float conversion of `host`, when needed
  rcdl::TextRecognizer task;

  template <typename Dict>
  BoundRecognizer(rcdl::Engine& e, const std::string& model_input, const std::string& fit,
                  float in_scale, float in_shift, const std::string& backend_name, Dict&& dict,
                  const rcdl::OcrRecConfig& cfg, int output_index)
      : engine(&e),
        in_fmt(formatFromName(model_input)),
        scale(in_scale),
        shift(in_shift),
        stretch(fit == "stretch"),
        backend(backendFromName(backend_name)),
        task(e, std::forward<Dict>(dict), cfg, output_index) {
    if (fit != "stretch" && fit != "letterbox") {
      throw std::invalid_argument("unknown fit mode: " + fit);
    }
    const rknn_tensor_attr& a = e.inputAttr(0);
    if (a.n_dims != 4u || a.fmt != RKNN_TENSOR_NHWC) {
      throw std::invalid_argument("TextRecognizer: input 0 is not a 4-D NHWC image tensor");
    }
    in_h = static_cast<int>(a.dims[1]);
    in_w = static_cast<int>(a.dims[2]);
    const std::size_t bytes = rcdl::imageBytes(in_fmt, in_w, in_h);
    const std::size_t channels = static_cast<std::size_t>(a.dims[3]);
    if (in_w <= 0 || in_h <= 0 ||
        bytes != static_cast<std::size_t>(in_w) * static_cast<std::size_t>(in_h) * channels) {
      throw std::invalid_argument("TextRecognizer: input 0 has " + std::to_string(channels) +
                                  " channels, which " + std::string(pyFormatName(in_fmt)) +
                                  " does not carry");
    }
    const rknn_tensor_type t = e.inputType(0);
    if (t == RKNN_TENSOR_FLOAT32) {
      float_input = true;
      floats.resize(bytes);
    } else if (t != RKNN_TENSOR_UINT8) {
      throw std::invalid_argument("TextRecognizer: input 0 takes " +
                                  std::string(rcdl::dtypeName(t)) +
                                  "; only u8 and f32 image inputs are supported");
    }
    host.resize(bytes);
  }

  /// Preproc + infer. Call with the GIL released: this is all hardware bar the
  /// float conversion.
  void run(const rcdl::ImageView& src) {
    rcdl::ImageView dst = rcdl::hostView(host.data(), in_w, in_h, in_fmt);
    dst.size = host.size();
    if (stretch) {
      rcdl::resize(dst, src, backend, rcdl::YuvRange::kStudioToFull, &last_backend);
    } else {
      rcdl::letterbox(dst, src, pad, backend, rcdl::YuvRange::kStudioToFull, &last_backend);
    }
    if (float_input) {
      for (std::size_t i = 0; i < host.size(); ++i) {
        floats[i] = static_cast<float>(host[i]) * scale + shift;
      }
      engine->setInput(0, floats.data(), floats.size() * sizeof(float));
    } else {
      engine->setInput(0, host.data(), host.size());
    }
    engine->infer();
  }
};

/// The Engine-bound direction classifier, which cannot be a plain BoundTask
/// because its input is not a letterbox.
///
/// PP-OCR fits a line crop to the model's HEIGHT, caps the width, anchors it at
/// the top-left and pads the rest — see ocrLineFitWidth() for what feeding it a
/// centred letterbox instead costs (16/16 orientations right becomes 9/16). The
/// resize therefore goes into a LEFT SUB-VIEW of a host scratch buffer, whose
/// remainder is set to the pad value, and the result is handed over with
/// setInput. A host buffer rather than the NPU's tensor because the padding is a
/// CPU write, and a CPU write into a buffer RGA is also writing is exactly the
/// race docs/RGA.md 3.1 is about.
struct BoundAngleClassifier {
  rcdl::Engine* engine;
  int in_w = 0;
  int in_h = 0;
  rcdl::PixelFormat in_fmt;
  std::uint8_t pad;
  rcdl::PreprocBackend backend;
  rcdl::PreprocBackend last_backend = rcdl::PreprocBackend::Auto;
  int last_fit_w = 0;
  std::vector<std::uint8_t> host;  ///< packed WxHxC scratch, reused per call
  rcdl::TextAngleClassifier task;

  BoundAngleClassifier(rcdl::Engine& e, const std::string& model_input, std::uint8_t pad_value,
                       const std::string& backend_name, float thresh, int output_index)
      : engine(&e),
        in_fmt(formatFromName(model_input)),
        pad(pad_value),
        backend(backendFromName(backend_name)),
        task(e, thresh, output_index) {
    const rknn_tensor_attr& a = e.inputAttr(0);
    if (a.n_dims != 4u || a.fmt != RKNN_TENSOR_NHWC) {
      throw std::invalid_argument("TextAngleClassifier: input 0 is not a 4-D NHWC image tensor");
    }
    in_h = static_cast<int>(a.dims[1]);
    in_w = static_cast<int>(a.dims[2]);
    const std::size_t bytes = rcdl::imageBytes(in_fmt, in_w, in_h);
    const std::size_t channels = static_cast<std::size_t>(a.dims[3]);
    if (in_w <= 0 || in_h <= 0 ||
        bytes != static_cast<std::size_t>(in_w) * static_cast<std::size_t>(in_h) * channels) {
      throw std::invalid_argument("TextAngleClassifier: input 0 has " + std::to_string(channels) +
                                  " channels, which " + std::string(pyFormatName(in_fmt)) +
                                  " does not carry");
    }
    if (e.inputType(0) != RKNN_TENSOR_UINT8) {
      throw std::invalid_argument("TextAngleClassifier: input 0 takes " +
                                  std::string(rcdl::dtypeName(e.inputType(0))) +
                                  "; this head expects a quantized u8 image input");
    }
    host.resize(bytes);
  }

  /// Preproc + infer. Call with the GIL released.
  void run(const rcdl::ImageView& src) {
    last_fit_w = rcdl::ocrLineFitWidth(src.width, src.height, in_w, in_h);
    rcdl::ImageView dst = rcdl::hostView(host.data(), in_w, in_h, in_fmt);
    dst.size = host.size();
    if (last_fit_w < in_w) {
      // Pad first, then resize: both are host writes on a buffer no hardware
      // owns, and painting the whole canvas is cheaper to get right than
      // painting the leftover strip row by row.
      std::fill(host.begin(), host.end(), pad);
      dst.width = last_fit_w;  // the left sub-view; the row stride stays in_w
      dst.wstride = in_w;
    }
    rcdl::resize(dst, src, backend, rcdl::YuvRange::kStudioToFull, &last_backend);
    engine->setInput(0, host.data(), host.size());
    engine->infer();
  }
};

// An Engine-bound head that only postprocesses (Segmenter, DepthEstimator,
// InstanceSegmenter) plus the preprocessing it needs to be usable from Python:
// RGA letterboxes the source straight into the NPU's input tensor, exactly as
// DetectionPipeline does, so process() is one hardware chain with no host copy.
template <typename Task>
struct BoundTask {
  rcdl::Engine* engine;
  rcdl::ImageView input;  ///< the NPU's own input tensor, resolved once
  std::uint8_t pad;
  rcdl::PreprocBackend backend;
  rcdl::PreprocBackend last_backend = rcdl::PreprocBackend::Auto;
  rcdl::LetterboxInfo last_lb;
  Task task;

  template <typename... Args>
  BoundTask(rcdl::Engine& e, const std::string& model_input, std::uint8_t pad_value,
            const std::string& backend_name, Args&&... args)
      : engine(&e),
        input(rcdl::engineInputView(e, 0, formatFromName(model_input))),
        pad(pad_value),
        backend(backendFromName(backend_name)),
        task(e, std::forward<Args>(args)...) {}

  /// Preproc + infer. Call with the GIL released: this is all hardware.
  void run(const rcdl::ImageView& src) {
    last_lb = rcdl::letterbox(input, src, pad, backend, rcdl::YuvRange::kStudioToFull,
                              &last_backend);
    engine->infer();
  }
};
}  // namespace

NB_MODULE(rcdl_py, m) {
  m.doc() = "RCDL — RKNPU inference & media library (compiled core)";
  m.attr("__version__") = RCDL_VERSION_STRING;

  nb::enum_<rcdl::NpuCore>(m, "NpuCore")
      .value("AUTO", rcdl::NpuCore::Auto)
      .value("CORE_0", rcdl::NpuCore::Core0)
      .value("CORE_1", rcdl::NpuCore::Core1)
      .value("CORE_2", rcdl::NpuCore::Core2)
      .value("CORE_0_1", rcdl::NpuCore::Core01)
      .value("CORE_0_1_2", rcdl::NpuCore::Core012)
      .value("ALL", rcdl::NpuCore::All);

  nb::class_<rcdl::Engine>(m, "Engine")
      .def(
          "__init__",
          [](rcdl::Engine* self, const std::string& path, rcdl::NpuCore core,
             std::uint32_t init_flags) {
            rcdl::Engine::Options o;
            o.core = core;
            o.init_flags = init_flags;
            new (self) rcdl::Engine(path, o);
          },
          "path"_a, "core"_a = rcdl::NpuCore::Auto, "init_flags"_a = 0u)
      .def("dup", &rcdl::Engine::dup, "core"_a = rcdl::NpuCore::Auto)
      .def_prop_ro("path", &rcdl::Engine::path)
      .def_prop_ro("core", &rcdl::Engine::core)
      .def_prop_ro("num_inputs", &rcdl::Engine::numInputs)
      .def_prop_ro("num_outputs", &rcdl::Engine::numOutputs)
      .def("input_shape", &rcdl::Engine::inputShape, "i"_a)
      .def("output_shape", &rcdl::Engine::outputShape, "i"_a)
      .def("input_name", &rcdl::Engine::inputName, "i"_a)
      .def("output_name", &rcdl::Engine::outputName, "i"_a)
      .def(
          "input_dtype", [](const rcdl::Engine& e, int i) { return rcdl::dtypeName(e.inputType(i)); },
          "i"_a, "dtype the caller provides for input i ('u8' for quantized image models, 'f32' for float models)")
      .def(
          "output_dtype",
          [](const rcdl::Engine& e, int i) { return rcdl::dtypeName(e.outputType(i)); }, "i"_a)
      .def(
          "input_format",
          [](const rcdl::Engine& e, int i) { return get_format_string(e.inputFormat(i)); }, "i"_a)
      .def(
          "output_quant",
          [](const rcdl::Engine& e, int i) {
            const auto& a = e.outputAttr(i);
            return nb::make_tuple(static_cast<int>(a.qnt_type), a.zp, a.scale, static_cast<int>(a.fl));
          },
          "i"_a, "(qnt_type, zp, scale, fl) of output i")
      .def("input_bytes", &rcdl::Engine::inputBytes, "i"_a)
      .def("input_packed_bytes", &rcdl::Engine::inputPackedBytes, "i"_a)
      .def("input_width_stride", &rcdl::Engine::inputWidthStride, "i"_a)
      .def("input_fd", &rcdl::Engine::inputFd, "i"_a)
      .def("output_bytes", &rcdl::Engine::outputBytes, "i"_a)
      .def("output_packed_bytes", &rcdl::Engine::outputPackedBytes, "i"_a)
      .def("output_fd", &rcdl::Engine::outputFd, "i"_a)
      .def("set_input", &setInputFromArray, "i"_a, "array"_a,
           "Copy a C-contiguous array (packed or device-strided byte count) into input i")
      .def(
          "infer", [](rcdl::Engine& e) { nb::gil_scoped_release nogil; e.infer(); },
          "Run one inference (blocking, GIL released)")
      .def(
          "infer_async", [](rcdl::Engine& e) { nb::gil_scoped_release nogil; e.inferAsync(); })
      .def(
          "wait",
          [](rcdl::Engine& e, int timeout_ms) {
            nb::gil_scoped_release nogil;
            e.wait(timeout_ms);
          },
          "timeout_ms"_a = 0)
      .def("output_float", &outputFloat, "i"_a,
           "Output i dequantized to a float32 numpy array of the model's shape")
      .def(
          "output_raw",
          [](const rcdl::Engine& e, int i) {
            return nb::bytes(static_cast<const char*>(e.outputData(i)), e.outputBytes(i));
          },
          "i"_a, "Raw output buffer (model encoding, device layout)")
      .def("last_run_micros", &rcdl::Engine::lastRunMicros)
      .def("perf_detail", &rcdl::Engine::perfDetail)
      .def("sdk_version", &rcdl::Engine::sdkVersion)
      .def("driver_version", &rcdl::Engine::driverVersion);

  nb::enum_<rcdl::DmaBuf::Heap>(m, "DmaHeap")
      .value("SYSTEM", rcdl::DmaBuf::Heap::System)
      .value("SYSTEM_UNCACHED", rcdl::DmaBuf::Heap::SystemUncached)
      .value("CMA", rcdl::DmaBuf::Heap::Cma)
      .value("CMA_UNCACHED", rcdl::DmaBuf::Heap::CmaUncached);

  nb::class_<rcdl::DmaBuf>(m, "DmaBuf")
      .def_static(
          "alloc",
          [](std::size_t size, rcdl::DmaBuf::Heap heap) {
            return std::make_unique<rcdl::DmaBuf>(rcdl::DmaBuf::alloc(size, heap));
          },
          "size"_a, "heap"_a = rcdl::DmaBuf::Heap::System)
      .def_prop_ro("fd", &rcdl::DmaBuf::fd)
      .def_prop_ro("size", &rcdl::DmaBuf::size)
      .def("sync_start", &rcdl::DmaBuf::syncStart, "read"_a = true, "write"_a = true)
      .def("sync_end", &rcdl::DmaBuf::syncEnd, "read"_a = true, "write"_a = true)
      .def(
          "write",
          [](rcdl::DmaBuf& b, nb::bytes data, std::size_t offset) {
            if (offset + data.size() > b.size()) throw std::out_of_range("DmaBuf.write overflow");
            std::memcpy(static_cast<char*>(b.data()) + offset, data.c_str(), data.size());
          },
          "data"_a, "offset"_a = 0)
      .def(
          "read",
          [](rcdl::DmaBuf& b, std::size_t n, std::size_t offset) {
            if (offset + n > b.size()) throw std::out_of_range("DmaBuf.read overflow");
            return nb::bytes(static_cast<const char*>(b.data()) + offset, n);
          },
          "n"_a, "offset"_a = 0);

  m.def("dequantize", [](nb::bytes raw, const std::string& dtype, int qnt_type, int zp, float scale,
                         int fl) {
    rknn_tensor_attr a{};
    if (dtype == "f32") a.type = RKNN_TENSOR_FLOAT32;
    else if (dtype == "f16") a.type = RKNN_TENSOR_FLOAT16;
    else if (dtype == "i8") a.type = RKNN_TENSOR_INT8;
    else if (dtype == "u8") a.type = RKNN_TENSOR_UINT8;
    else if (dtype == "i16") a.type = RKNN_TENSOR_INT16;
    else if (dtype == "u16") a.type = RKNN_TENSOR_UINT16;
    else if (dtype == "i32") a.type = RKNN_TENSOR_INT32;
    else if (dtype == "u32") a.type = RKNN_TENSOR_UINT32;
    else if (dtype == "i64") a.type = RKNN_TENSOR_INT64;
    else if (dtype == "bool") a.type = RKNN_TENSOR_BOOL;
    else throw std::invalid_argument("dequantize: unknown dtype " + dtype);
    a.qnt_type = static_cast<rknn_tensor_qnt_type>(qnt_type);
    a.zp = zp;
    a.scale = scale;
    a.fl = static_cast<std::int8_t>(fl);
    const std::size_t elem = rcdl::elementSize(a.type);
    const std::size_t n = raw.size() / elem;
    float* buf = new float[n];
    rcdl::dequantizeToFloat(a, raw.c_str(), buf, n);
    nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    std::size_t shape[1] = {n};
    return nb::ndarray<nb::numpy, float>(buf, 1, shape, owner);
  }, "raw"_a, "dtype"_a, "qnt_type"_a = 0, "zp"_a = 0, "scale"_a = 1.0f, "fl"_a = 0,
     "Dequantize raw tensor bytes the way Engine.output_float does (test hook; needs no NPU)");

  // --- geometry ---------------------------------------------------------------
  m.def(
      "compute_letterbox",
      [](int src_w, int src_h, int dst_w, int dst_h, bool center_pad) {
        return lbToTuple(rcdl::computeLetterbox(src_w, src_h, dst_w, dst_h, center_pad));
      },
      "src_w"_a, "src_h"_a, "dst_w"_a, "dst_h"_a, "center_pad"_a = true,
      "(scale, pad_x, pad_y, src_w, src_h, dst_w, dst_h) for an aspect-preserving fit");

  // --- preprocessing ----------------------------------------------------------
  m.def("rga_available", &rcdl::rgaAvailable, "Is the RGA 2-D engine usable in this build/board?");
  m.def("rga_version", &rcdl::rgaVersion);

  m.def(
      "letterbox",
      [](const Contig& src, int src_w, int src_h, const std::string& src_fmt, int dst_w,
         int dst_h, const std::string& dst_fmt, std::uint8_t pad, const std::string& backend,
         bool studio_range, int src_wstride, int src_hstride) {
        const rcdl::ImageView sv = viewFromArray(src, src_w, src_h, src_fmt, src_wstride, src_hstride);
        const rcdl::PixelFormat df = formatFromName(dst_fmt);
        const int dws = rcdl::alignUp(dst_w, rcdl::strideAlign(df));
        const std::size_t nbytes = rcdl::imageBytes(df, dws, dst_h);
        auto* buf = new std::uint8_t[nbytes];
        nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<std::uint8_t*>(p); });
        rcdl::ImageView dv = rcdl::hostView(buf, dst_w, dst_h, df, dws, dst_h);
        dv.size = nbytes;
        rcdl::PreprocBackend used = rcdl::PreprocBackend::Auto;
        rcdl::LetterboxInfo lb;
        {
          nb::gil_scoped_release nogil;
          lb = rcdl::letterbox(dv, sv, pad, backendFromName(backend),
                               studio_range ? rcdl::YuvRange::kStudioToFull : rcdl::YuvRange::kAsIs,
                               &used);
        }
        std::size_t shape[1] = {nbytes};
        auto arr = nb::ndarray<nb::numpy, std::uint8_t>(buf, 1, shape, owner);
        return nb::make_tuple(arr, lbToTuple(lb), std::string(rcdl::backendName(used)), dws);
      },
      "src"_a, "src_w"_a, "src_h"_a, "src_fmt"_a, "dst_w"_a, "dst_h"_a, "dst_fmt"_a = "rgb888",
      "pad"_a = std::uint8_t(114), "backend"_a = "auto", "studio_range"_a = true,
      "src_wstride"_a = 0, "src_hstride"_a = 0,
      "Letterbox a uint8 image buffer; returns (flat_dst_bytes, letterbox, backend, dst_wstride)");

  m.def(
      "cvt_color",
      [](const Contig& src, int w, int h, const std::string& src_fmt, const std::string& dst_fmt,
         const std::string& backend, bool studio_range, int src_wstride, int src_hstride) {
        const rcdl::ImageView sv = viewFromArray(src, w, h, src_fmt, src_wstride, src_hstride);
        const rcdl::PixelFormat df = formatFromName(dst_fmt);
        const int dws = rcdl::alignUp(w, rcdl::strideAlign(df));
        const std::size_t nbytes = rcdl::imageBytes(df, dws, h);
        auto* buf = new std::uint8_t[nbytes];
        nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<std::uint8_t*>(p); });
        rcdl::ImageView dv = rcdl::hostView(buf, w, h, df, dws, h);
        dv.size = nbytes;
        rcdl::PreprocBackend used = rcdl::PreprocBackend::Auto;
        {
          nb::gil_scoped_release nogil;
          rcdl::cvtColor(dv, sv, backendFromName(backend),
                         studio_range ? rcdl::YuvRange::kStudioToFull : rcdl::YuvRange::kAsIs,
                         &used);
        }
        std::size_t shape[1] = {nbytes};
        auto arr = nb::ndarray<nb::numpy, std::uint8_t>(buf, 1, shape, owner);
        return nb::make_tuple(arr, std::string(rcdl::backendName(used)), dws);
      },
      "src"_a, "w"_a, "h"_a, "src_fmt"_a, "dst_fmt"_a, "backend"_a = "auto",
      "studio_range"_a = true, "src_wstride"_a = 0, "src_hstride"_a = 0,
      "Colour-convert a uint8 image; returns (flat_dst_bytes, backend, dst_wstride)");

  // --- detection post-processing ----------------------------------------------
  nb::class_<rcdl::Detection>(m, "Detection")
      .def(nb::init<>())
      .def_rw("x1", &rcdl::Detection::x1)
      .def_rw("y1", &rcdl::Detection::y1)
      .def_rw("x2", &rcdl::Detection::x2)
      .def_rw("y2", &rcdl::Detection::y2)
      .def_rw("score", &rcdl::Detection::score)
      .def_rw("class_id", &rcdl::Detection::class_id)
      .def("__repr__", [](const rcdl::Detection& d) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "Detection(cls=%d score=%.3f box=[%.1f,%.1f,%.1f,%.1f])",
                      d.class_id, d.score, d.x1, d.y1, d.x2, d.y2);
        return std::string(buf);
      });

  m.def("coco_class_name", &rcdl::cocoClassName, "class_id"_a);
  m.def("coco_class_names", &rcdl::cocoClassNames);

  m.def(
      "nms",
      [](const Contig& boxes, float iou_thresh, int max_dets) {
        if (boxes.ndim() != 2 || boxes.shape(1) != 6) {
          throw std::invalid_argument("nms: expected an (N, 6) [x1,y1,x2,y2,score,class] array");
        }
        const float* p = floatData(boxes, "nms");
        std::vector<rcdl::Detection> dets(boxes.shape(0));
        for (std::size_t i = 0; i < boxes.shape(0); ++i) {
          const float* r = p + i * 6;
          dets[i] = {r[0], r[1], r[2], r[3], r[4], static_cast<int>(r[5])};
        }
        return rcdl::nms(dets, iou_thresh, max_dets);
      },
      "boxes"_a, "iou_thresh"_a = 0.45f, "max_dets"_a = 300,
      "Per-class greedy NMS over an (N,6) float32 array; returns kept row indices");

  m.def(
      "decode",
      [](const Contig& tensor, const LbTuple& lb, int num_classes, float conf_thresh,
         float iou_thresh, int max_dets, bool channels_first, bool apply_sigmoid, bool has_obj) {
        rcdl::DetectConfig cfg;
        cfg.num_classes = num_classes;
        cfg.conf_thresh = conf_thresh;
        cfg.iou_thresh = iou_thresh;
        cfg.max_dets = max_dets;
        cfg.channels_first = channels_first;
        cfg.apply_sigmoid = apply_sigmoid;
        cfg.layout = has_obj ? rcdl::DecodeLayout::kYoloV5 : rcdl::DecodeLayout::kYoloV8;
        const float* data = floatData(tensor, "decode");
        std::vector<int> shape;
        for (std::size_t d = 0; d < tensor.ndim(); ++d) shape.push_back(static_cast<int>(tensor.shape(d)));
        return rcdl::decode(data, shape, cfg, lbFromTuple(lb));
      },
      "tensor"_a, "letterbox"_a, "num_classes"_a = 80, "conf_thresh"_a = 0.25f,
      "iou_thresh"_a = 0.45f, "max_dets"_a = 300, "channels_first"_a = true,
      "apply_sigmoid"_a = false, "has_obj"_a = false,
      "Decode a fused single-tensor YOLO head (float32) into Detections");

  m.def(
      "decode_yolo_ltrb",
      [](const std::vector<Contig>& cls, const std::vector<Contig>& box,
         const std::vector<std::pair<int, int>>& grid_hw, const std::vector<int>& strides,
         const LbTuple& lb, int num_classes, float conf_thresh, float iou_thresh, int max_dets,
         int reg_max, bool channels_first, bool apply_sigmoid) {
        rcdl::YoloLtrbConfig cfg;
        cfg.num_classes = num_classes;
        cfg.conf_thresh = conf_thresh;
        cfg.iou_thresh = iou_thresh;
        cfg.max_dets = max_dets;
        cfg.strides = strides;
        cfg.reg_max = reg_max;
        cfg.channels_first = channels_first;
        cfg.apply_sigmoid = apply_sigmoid;
        // The decoder indexes up to (nc-1)*H*W and (4*reg_max-1)*H*W from these
        // pointers using the caller's grid/class/reg_max numbers. Nothing in
        // numpy ties those to the arrays' real sizes, so an inconsistent call
        // would read far out of bounds — check here, the way the C++
        // YoloLtrbDetector checks before it decodes.
        if (cls.size() != box.size() || cls.size() != grid_hw.size() ||
            cls.size() != strides.size()) {
          throw std::invalid_argument(
              "decode_yolo_ltrb: cls, box, grid_hw and strides must have the same length");
        }
        const int box_ch = reg_max > 0 ? 4 * reg_max : 4;
        std::vector<const float*> cp, bp;
        cp.reserve(cls.size());
        bp.reserve(box.size());
        for (std::size_t i = 0; i < cls.size(); ++i) {
          const std::size_t cells = static_cast<std::size_t>(grid_hw[i].first) *
                                    static_cast<std::size_t>(grid_hw[i].second);
          const std::size_t need_cls = cells * static_cast<std::size_t>(num_classes);
          const std::size_t need_box = cells * static_cast<std::size_t>(box_ch);
          if (elemCount(cls[i]) < need_cls || elemCount(box[i]) < need_box) {
            throw std::invalid_argument(
                "decode_yolo_ltrb: scale " + std::to_string(i) + " needs " +
                std::to_string(need_cls) + " cls and " + std::to_string(need_box) +
                " box elements for a " + std::to_string(grid_hw[i].first) + "x" +
                std::to_string(grid_hw[i].second) + " grid, got " +
                std::to_string(elemCount(cls[i])) + " and " + std::to_string(elemCount(box[i])));
          }
          cp.push_back(floatData(cls[i], "decode_yolo_ltrb cls"));
          bp.push_back(floatData(box[i], "decode_yolo_ltrb box"));
        }
        return rcdl::decodeYoloLtrb(cp, bp, grid_hw, cfg, lbFromTuple(lb));
      },
      "cls"_a, "box"_a, "grid_hw"_a, "strides"_a, "letterbox"_a, "num_classes"_a = 80,
      "conf_thresh"_a = 0.25f, "iou_thresh"_a = 0.45f, "max_dets"_a = 300, "reg_max"_a = 0,
      "channels_first"_a = true, "apply_sigmoid"_a = false,
      "Decode the anchor-free LTRB multi-scale head from per-scale float32 cls/box buffers");

  // --- detection pipeline -------------------------------------------------------
  nb::class_<rcdl::DetectionPipeline>(m, "DetectionPipeline")
      .def(
          "__init__",
          [](rcdl::DetectionPipeline* self, nb::handle engine_arg, const std::string& model_input,
             float conf_thresh, float iou_thresh, int max_dets, int num_classes,
             bool apply_sigmoid, std::uint8_t pad, const std::string& backend) {
            rcdl::Engine& engine = engineFrom(engine_arg);
            rcdl::PipelineConfig cfg;
            cfg.model_input = formatFromName(model_input);
            cfg.pad_value = pad;
            cfg.backend = backendFromName(backend);
            cfg.detect.conf_thresh = cfg.ltrb.conf_thresh = conf_thresh;
            cfg.detect.iou_thresh = cfg.ltrb.iou_thresh = iou_thresh;
            cfg.detect.max_dets = cfg.ltrb.max_dets = max_dets;
            cfg.detect.num_classes = cfg.ltrb.num_classes = num_classes;
            cfg.detect.apply_sigmoid = cfg.ltrb.apply_sigmoid = apply_sigmoid;
            new (self) rcdl::DetectionPipeline(engine, cfg);
          },
          "engine"_a, "model_input"_a = "rgb888", "conf_thresh"_a = 0.25f, "iou_thresh"_a = 0.45f,
          "max_dets"_a = 300, "num_classes"_a = 80, "apply_sigmoid"_a = false,
          "pad"_a = std::uint8_t(114), "backend"_a = "auto", nb::keep_alive<1, 2>())
      .def(
          "process",
          [](rcdl::DetectionPipeline& p, const Contig& img, int w, int h,
             const std::string& fmt, int wstride, int hstride) {
            const rcdl::ImageView v = viewFromArray(img, w, h, fmt, wstride, hstride);
            nb::gil_scoped_release nogil;  // preproc + NPU infer + decode
            return p.process(v);
          },
          "image"_a, "w"_a, "h"_a, "fmt"_a = "bgr888", "wstride"_a = 0, "hstride"_a = 0,
          "Preprocess, infer and decode one frame; returns Detections in source pixels")
      .def(
          "process_frame",
          [](rcdl::DetectionPipeline& p, const rcdl::VideoFrame& f) {
            nb::gil_scoped_release nogil;  // RGA reads the decoder's dma-buf directly
            return p.process(f.view());
          },
          "frame"_a,
          "Detect on a decoded VideoFrame without copying it out of the VPU's buffer")
      .def_prop_ro("letterbox", [](const rcdl::DetectionPipeline& p) {
        return lbToTuple(p.lastLetterbox());
      })
      .def_prop_ro("backend", [](const rcdl::DetectionPipeline& p) {
        return std::string(rcdl::backendName(p.lastBackend()));
      })
      .def_prop_ro("head", [](const rcdl::DetectionPipeline& p) {
        return std::string(rcdl::headName(p.head()));
      })
      .def_prop_ro("head_layout", [](const rcdl::DetectionPipeline& p) {
        const auto* d = p.decoder().ltrb();
        return d ? d->layout().describe() : std::string("(single-tensor head)");
      })
      .def("reset_profile", &rcdl::DetectionPipeline::resetProfile)
      .def_prop_ro("profile", [](const rcdl::DetectionPipeline& p) {
        const auto& s = p.profile();
        return nb::make_tuple(s.preprocPerFrame(), s.inferPerFrame(), s.postprocPerFrame(),
                              s.frames);
      }, "(preproc_ms, infer_ms, postproc_ms, frames) averaged per frame");

  // --- media: VPU codecs --------------------------------------------------------
  m.def("mpp_available", &rcdl::mppAvailable,
        "Is the VPU (MPP) usable in this build/board?");

  nb::class_<rcdl::VideoFrame>(m, "VideoFrame")
      .def_prop_ro("valid", &rcdl::VideoFrame::valid)
      .def_prop_ro("width", &rcdl::VideoFrame::width)
      .def_prop_ro("height", &rcdl::VideoFrame::height)
      .def_prop_ro("fd", &rcdl::VideoFrame::fd,
                   "dma-buf fd the VPU decoded into (-1 if none) — the zero-copy handle")
      .def_prop_ro("pts_us", &rcdl::VideoFrame::ptsUs)
      .def_prop_ro("index", &rcdl::VideoFrame::index)
      .def_prop_ro("format",
                   [](const rcdl::VideoFrame& f) { return std::string(pyFormatName(f.format())); })
      .def_prop_ro("width_stride", [](const rcdl::VideoFrame& f) { return f.view().effWStride(); })
      .def_prop_ro("height_stride", [](const rcdl::VideoFrame& f) { return f.view().effHStride(); })
      .def("release", &rcdl::VideoFrame::reset,
           "Return the buffer to the decoder pool now; holding frames stalls decoding")
      .def("__repr__", [](const rcdl::VideoFrame& f) { return f.describe(); })
      .def(
          "to_numpy",
          [](rcdl::VideoFrame& f, bool keep_stride) {
            // Always a COPY: the decoder needs its buffer back within a few
            // frames, so handing numpy a view would alias recycled memory.
            //
            // By default the VPU's stride padding is removed as we copy, because
            // hor_stride/ver_stride are aligned up from the display size and an
            // array shaped by `width` over stride-padded rows is the classic way
            // to get a sheared picture. `keep_stride=True` returns the raw device
            // layout for code that wants to hand the bytes back to hardware.
            const rcdl::ImageView& v = f.view();
            const int w = v.width, h = v.height;
            const std::size_t src_row = v.rowBytes();
            const int bpp = rcdl::bytesPerPixel(v.format);
            const bool yuv = bpp == 0;
            const std::size_t dst_row =
                keep_stride ? src_row : static_cast<std::size_t>(w) * (yuv ? 1 : bpp);
            const int rows = keep_stride ? v.effHStride() : h;
            const int chroma_rows = keep_stride ? v.effHStride() / 2 : h / 2;
            const std::size_t n =
                dst_row * static_cast<std::size_t>(rows) +
                (yuv ? dst_row * static_cast<std::size_t>(chroma_rows) : 0);

            const std::uint8_t* src = f.beginCpuRead();
            auto* buf = new std::uint8_t[n];
            for (int r = 0; r < rows; ++r) {
              std::memcpy(buf + static_cast<std::size_t>(r) * dst_row,
                          src + static_cast<std::size_t>(r) * src_row, dst_row);
            }
            if (yuv) {
              const std::uint8_t* uv = src + v.uvOffset();
              std::uint8_t* dst = buf + dst_row * static_cast<std::size_t>(rows);
              for (int r = 0; r < chroma_rows; ++r) {
                std::memcpy(dst + static_cast<std::size_t>(r) * dst_row,
                            uv + static_cast<std::size_t>(r) * src_row, dst_row);
              }
            }
            f.endCpuRead();

            nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<std::uint8_t*>(p); });
            const int out_w = static_cast<int>(keep_stride ? v.effWStride() : w);
            if (yuv) {
              std::size_t shape[2] = {static_cast<std::size_t>(rows + chroma_rows),
                                      static_cast<std::size_t>(out_w)};
              return nb::ndarray<nb::numpy, std::uint8_t>(buf, 2, shape, owner);
            }
            if (bpp == 1) {
              std::size_t shape[2] = {static_cast<std::size_t>(rows),
                                      static_cast<std::size_t>(out_w)};
              return nb::ndarray<nb::numpy, std::uint8_t>(buf, 2, shape, owner);
            }
            std::size_t shape[3] = {static_cast<std::size_t>(rows),
                                    static_cast<std::size_t>(out_w),
                                    static_cast<std::size_t>(bpp)};
            return nb::ndarray<nb::numpy, std::uint8_t>(buf, 3, shape, owner);
          },
          "keep_stride"_a = false,
          "Copy the frame out as a numpy array: (H*3//2, W) for NV12, (H, W, C) for "
          "packed formats. Stride padding is removed unless keep_stride is set.")
      .def(
          "letterbox",
          [](const rcdl::VideoFrame& f, int dst_w, int dst_h, const std::string& dst_fmt,
             std::uint8_t pad, const std::string& backend, bool studio_range) {
            // Reads the VPU's buffer by fd — the frame is never copied to the
            // host on the way in, which is the whole point of the media layer.
            const rcdl::PixelFormat df = formatFromName(dst_fmt);
            const int dws = rcdl::alignUp(dst_w, rcdl::strideAlign(df));
            const std::size_t nbytes = rcdl::imageBytes(df, dws, dst_h);
            std::unique_ptr<std::uint8_t[]> scratch(new std::uint8_t[nbytes]);
            std::uint8_t* buf = scratch.get();
            rcdl::ImageView dv = rcdl::hostView(buf, dst_w, dst_h, df, dws, dst_h);
            dv.size = nbytes;
            rcdl::PreprocBackend used = rcdl::PreprocBackend::Auto;
            rcdl::LetterboxInfo lb;
            {
              nb::gil_scoped_release nogil;
              lb = rcdl::letterbox(dv, f.view(), pad, backendFromName(backend),
                                   studio_range ? rcdl::YuvRange::kStudioToFull
                                                : rcdl::YuvRange::kAsIs,
                                   &used);
            }
            // Shape it here rather than handing back flat bytes plus a stride:
            // this is the convenience entry point, and `rcdl.letterbox()` in the
            // Python wrapper returns a shaped array too. `dws` may exceed dst_w
            // (strideAlign), so drop the padding as we shape.
            const int bpp = rcdl::bytesPerPixel(df);
            const bool yuv = bpp == 0;
            const std::size_t out_row =
                static_cast<std::size_t>(dst_w) * (yuv ? 1 : bpp);
            const int out_rows = yuv ? dst_h * 3 / 2 : dst_h;
            auto* packed = new std::uint8_t[out_row * static_cast<std::size_t>(out_rows)];
            for (int r = 0; r < out_rows; ++r) {
              std::memcpy(packed + static_cast<std::size_t>(r) * out_row,
                          buf + static_cast<std::size_t>(r) * dv.rowBytes(), out_row);
            }
            scratch.reset();
            nb::capsule powner(packed,
                               [](void* q) noexcept { delete[] static_cast<std::uint8_t*>(q); });
            if (yuv || bpp == 1) {
              std::size_t shape[2] = {static_cast<std::size_t>(out_rows),
                                      static_cast<std::size_t>(dst_w)};
              return nb::make_tuple(nb::ndarray<nb::numpy, std::uint8_t>(packed, 2, shape, powner),
                                    lbToTuple(lb), std::string(rcdl::backendName(used)));
            }
            std::size_t shape[3] = {static_cast<std::size_t>(dst_h),
                                    static_cast<std::size_t>(dst_w),
                                    static_cast<std::size_t>(bpp)};
            return nb::make_tuple(nb::ndarray<nb::numpy, std::uint8_t>(packed, 3, shape, powner),
                                  lbToTuple(lb), std::string(rcdl::backendName(used)));
          },
          "dst_w"_a, "dst_h"_a, "dst_fmt"_a = "rgb888", "pad"_a = std::uint8_t(114),
          "backend"_a = "auto", "studio_range"_a = true,
          "Letterbox this frame out of the VPU's buffer without copying it first; "
          "returns (image, letterbox, backend)");

  nb::class_<rcdl::VideoDecoder>(m, "VideoDecoder")
      .def(
          "__init__",
          [](rcdl::VideoDecoder* self, const std::string& codec, const std::string& fmt,
             bool split_parse, bool external_buffers, int buffer_count, int extra_buffers) {
            rcdl::VideoDecConfig c;
            c.codec = codecFromName(codec);
            c.format = formatFromName(fmt);
            c.split_parse = split_parse;
            c.external_buffers = external_buffers;
            c.buffer_count = buffer_count;
            c.extra_buffers = extra_buffers;
            new (self) rcdl::VideoDecoder(c);
          },
          "codec"_a = "h264", "format"_a = "nv12", "split_parse"_a = true,
          "external_buffers"_a = true, "buffer_count"_a = 0, "extra_buffers"_a = 4)
      .def(
          "feed",
          [](rcdl::VideoDecoder& d, nb::bytes data, std::uint64_t pts_us, int timeout_ms) {
            const auto* p = reinterpret_cast<const std::uint8_t*>(data.c_str());
            const std::size_t n = data.size();
            nb::gil_scoped_release nogil;
            return d.feed(p, n, pts_us, timeout_ms);
          },
          "data"_a, "pts_us"_a = 0, "timeout_ms"_a = 20,
          "Queue compressed bytes; False means the input queue is full — drain and retry")
      // keep_alive<0, 1>: the frame owns an MppFrame whose BUFFER belongs to the
      // decoder's group. Letting Python collect the decoder while a frame is
      // still referenced would destroy the group first, and the frame's
      // destructor would then release a buffer into freed MPP state.
      .def("receive", &receiveFrame<rcdl::VideoDecoder>, "timeout_ms"_a = 0,
           nb::keep_alive<0, 1>(), "Next decoded frame in display order, or None")
      .def(
          "flush",
          [](rcdl::VideoDecoder& d) {
            auto f = std::make_unique<rcdl::VideoFrame>();
            bool got;
            {
              nb::gil_scoped_release nogil;
              got = d.flush(*f);
            }
            return got ? std::move(f) : nullptr;
          },
          nb::keep_alive<0, 1>(),
          "End the stream, then drain what is left — call until it returns None")
      .def("feed_end_of_stream", &rcdl::VideoDecoder::feedEndOfStream)
      .def("reset", &rcdl::VideoDecoder::reset)
      .def_prop_ro("width", &rcdl::VideoDecoder::width)
      .def_prop_ro("height", &rcdl::VideoDecoder::height)
      .def_prop_ro("width_stride", &rcdl::VideoDecoder::widthStride)
      .def_prop_ro("height_stride", &rcdl::VideoDecoder::heightStride)
      .def_prop_ro("frames_decoded", &rcdl::VideoDecoder::framesDecoded)
      .def_prop_ro("end_of_stream", &rcdl::VideoDecoder::endOfStream)
      .def_prop_ro("using_external_buffers", &rcdl::VideoDecoder::usingExternalBuffers)
      .def_prop_ro("codec",
                   [](const rcdl::VideoDecoder& d) { return std::string(pyCodecName(d.codec())); });

  nb::class_<rcdl::VideoEncoder>(m, "VideoEncoder")
      .def(
          "__init__",
          [](rcdl::VideoEncoder* self, int width, int height, const std::string& codec,
             const std::string& fmt, int fps, int bitrate_kbps, int gop, const std::string& rc,
             int qp, int profile) {
            rcdl::VideoEncConfig c;
            c.codec = codecFromName(codec);
            c.width = width;
            c.height = height;
            c.format = formatFromName(fmt);
            c.fps = fps;
            c.bitrate_kbps = bitrate_kbps;
            c.gop = gop;
            c.rc = rcFromName(rc);
            c.qp = qp;
            c.profile = profile;
            new (self) rcdl::VideoEncoder(c);
          },
          "width"_a, "height"_a, "codec"_a = "h264", "format"_a = "nv12", "fps"_a = 30,
          "bitrate_kbps"_a = 4000, "gop"_a = 0, "rc"_a = "cbr", "qp"_a = 0, "profile"_a = 100)
      .def(
          "feed",
          [](rcdl::VideoEncoder& e, const Contig& img, int w, int h, const std::string& fmt,
             std::uint64_t pts_us, int timeout_ms, int wstride, int hstride) {
            const rcdl::ImageView v = viewFromArray(img, w, h, fmt, wstride, hstride);
            nb::gil_scoped_release nogil;
            return e.feed(v, pts_us, timeout_ms);
          },
          "image"_a, "w"_a, "h"_a, "fmt"_a = "nv12", "pts_us"_a = 0, "timeout_ms"_a = 20,
          "wstride"_a = 0, "hstride"_a = 0)
      .def(
          "feed_frame",
          [](rcdl::VideoEncoder& e, const rcdl::VideoFrame& f, std::uint64_t pts_us,
             int timeout_ms) {
            nb::gil_scoped_release nogil;
            // The decoder's dma-buf fd goes straight to the VPU: no copy.
            return e.feed(f.view(), pts_us, timeout_ms);
          },
          "frame"_a, "pts_us"_a = 0, "timeout_ms"_a = 20,
          "Encode a decoded frame in place (zero copy)")
      .def("receive", &receivePacket<rcdl::VideoEncoder>, "timeout_ms"_a = 0,
           "Next compressed packet as bytes, or None")
      .def(
          "flush",
          [](rcdl::VideoEncoder& e) -> nb::object {
            std::vector<std::uint8_t> out;
            bool got;
            {
              nb::gil_scoped_release nogil;
              got = e.flush(out);
            }
            if (!got) return nb::none();
            return nb::bytes(reinterpret_cast<const char*>(out.data()), out.size());
          })
      .def("feed_end_of_stream", &rcdl::VideoEncoder::feedEndOfStream)
      .def_prop_ro("extra_data",
                   [](const rcdl::VideoEncoder& e) {
                     const auto& x = e.extraData();
                     return nb::bytes(reinterpret_cast<const char*>(x.data()), x.size());
                   })
      .def_prop_ro("frames_encoded", &rcdl::VideoEncoder::framesEncoded)
      .def_prop_ro("width", &rcdl::VideoEncoder::width)
      .def_prop_ro("height", &rcdl::VideoEncoder::height)
      .def_prop_ro("codec",
                   [](const rcdl::VideoEncoder& e) { return std::string(pyCodecName(e.codec())); });

  nb::class_<rcdl::JpegEncoder>(m, "JpegEncoder")
      .def(
          "__init__",
          [](rcdl::JpegEncoder* self, int width, int height, const std::string& fmt, int quality) {
            new (self) rcdl::JpegEncoder(width, height, formatFromName(fmt), quality);
          },
          "width"_a, "height"_a, "format"_a = "nv12", "quality"_a = 80)
      .def(
          "encode",
          [](rcdl::JpegEncoder& j, const Contig& img, int w, int h, const std::string& fmt,
             int wstride, int hstride) {
            // The encoder already knows its size; -1 means "use it", so a caller
            // does not have to repeat what it passed to the constructor.
            if (w <= 0) w = j.width();
            if (h <= 0) h = j.height();
            const rcdl::ImageView v = viewFromArray(img, w, h, fmt, wstride, hstride);
            std::vector<std::uint8_t> out;
            {
              nb::gil_scoped_release nogil;
              out = j.encode(v);
            }
            return nb::bytes(reinterpret_cast<const char*>(out.data()), out.size());
          },
          "image"_a, "w"_a = -1, "h"_a = -1, "fmt"_a = "nv12", "wstride"_a = 0,
          "hstride"_a = 0)
      .def(
          "encode_frame",
          [](rcdl::JpegEncoder& j, const rcdl::VideoFrame& f) {
            std::vector<std::uint8_t> out;
            {
              nb::gil_scoped_release nogil;
              out = j.encode(f.view());  // the VPU reads the decoder's buffer in place
            }
            return nb::bytes(reinterpret_cast<const char*>(out.data()), out.size());
          },
          "frame"_a, "Encode a decoded frame in place (zero copy)");

  nb::class_<rcdl::JpegDecoder>(m, "JpegDecoder")
      .def(
          "__init__",
          [](rcdl::JpegDecoder* self, const std::string& fmt) {
            new (self) rcdl::JpegDecoder(formatFromName(fmt));
          },
          "format"_a = "nv12")
      .def(
          "decode",
          [](rcdl::JpegDecoder& j, nb::bytes data) -> std::unique_ptr<rcdl::VideoFrame> {
            auto f = std::make_unique<rcdl::VideoFrame>();
            const auto* p = reinterpret_cast<const std::uint8_t*>(data.c_str());
            const std::size_t n = data.size();
            bool got;
            {
              nb::gil_scoped_release nogil;
              got = j.decode(p, n, *f);
            }
            return got ? std::move(f) : nullptr;
          },
          "data"_a, nb::keep_alive<0, 1>(),
          "Decode a JPEG file into a dma-buf frame, or None if the VPU refused it");

  // --- async video detection pipeline -------------------------------------------
  //
  // The whole VPU -> RGA -> NPU path behind two calls. A Python driver that only
  // pumps bytes and takes detections runs at the C++ pipeline's speed, because
  // every stage — decode, letterbox, infer — is a C++ thread with the GIL
  // released; Python never sees a frame.
  nb::class_<rcdl::AsyncVideoDetectionPipeline>(m, "AsyncVideoDetectionPipeline")
      .def(
          "__init__",
          [](rcdl::AsyncVideoDetectionPipeline* self, nb::handle engine_arg,
             const std::string& codec, const std::string& model_input, float conf_thresh,
             float iou_thresh, int max_dets, int num_classes, bool apply_sigmoid,
             std::uint8_t pad, const std::string& backend, int workers, bool pin_cores,
             int reorder_depth, int queue_depth, bool external_buffers, int extra_buffers) {
            rcdl::Engine& engine = engineFrom(engine_arg);
            rcdl::PipelineConfig cfg;
            cfg.model_input = formatFromName(model_input);
            cfg.pad_value = pad;
            cfg.backend = backendFromName(backend);
            cfg.detect.conf_thresh = cfg.ltrb.conf_thresh = conf_thresh;
            cfg.detect.iou_thresh = cfg.ltrb.iou_thresh = iou_thresh;
            cfg.detect.max_dets = cfg.ltrb.max_dets = max_dets;
            cfg.detect.num_classes = cfg.ltrb.num_classes = num_classes;
            cfg.detect.apply_sigmoid = cfg.ltrb.apply_sigmoid = apply_sigmoid;
            rcdl::VideoDecConfig dec;
            dec.codec = codecFromName(codec);
            dec.external_buffers = external_buffers;
            dec.extra_buffers = extra_buffers;
            rcdl::VideoAsyncConfig vcfg;
            vcfg.async.workers = workers;
            vcfg.async.pin_cores = pin_cores;
            vcfg.async.reorder_depth = reorder_depth;
            vcfg.queue_depth = queue_depth;
            new (self) rcdl::AsyncVideoDetectionPipeline(engine, cfg, dec, vcfg);
          },
          "engine"_a, "codec"_a = "h264", "model_input"_a = "rgb888", "conf_thresh"_a = 0.25f,
          "iou_thresh"_a = 0.45f, "max_dets"_a = 300, "num_classes"_a = 80,
          "apply_sigmoid"_a = false, "pad"_a = std::uint8_t(114), "backend"_a = "auto",
          "workers"_a = 3, "pin_cores"_a = true, "reorder_depth"_a = 0, "queue_depth"_a = 2,
          "external_buffers"_a = true, "extra_buffers"_a = 4, nb::keep_alive<1, 2>())
      .def(
          "submit",
          [](rcdl::AsyncVideoDetectionPipeline& p, nb::bytes data, int timeout_ms) {
            const auto* d = reinterpret_cast<const std::uint8_t*>(data.c_str());
            const std::size_t n = data.size();
            nb::gil_scoped_release nogil;
            return p.submit(d, n, timeout_ms);
          },
          "data"_a, "timeout_ms"_a = 20,
          "Feed compressed bytes; False means back-pressure (drain and retry the SAME "
          "bytes) unless .finished")
      .def(
          "next",
          [](rcdl::AsyncVideoDetectionPipeline& p) -> nb::object {
            std::vector<rcdl::Detection> dets;
            bool got;
            {
              nb::gil_scoped_release nogil;
              got = p.next(dets);
            }
            return got ? nb::cast(dets) : nb::none();
          },
          "Next frame's detections in decode order, blocking; None once drained")
      .def(
          "try_next",
          [](rcdl::AsyncVideoDetectionPipeline& p) -> nb::object {
            std::vector<rcdl::Detection> dets;
            bool got;
            {
              nb::gil_scoped_release nogil;
              got = p.tryNext(dets);
            }
            return got ? nb::cast(dets) : nb::none();
          },
          "Non-blocking next(): None when nothing is ready yet")
      .def("finish", &rcdl::AsyncVideoDetectionPipeline::finish,
           nb::call_guard<nb::gil_scoped_release>(),
           "End of stream: flush the decoder's reorder tail, then drain with next()")
      .def_prop_ro("finished", &rcdl::AsyncVideoDetectionPipeline::finished)
      .def_prop_ro("pts_us", &rcdl::AsyncVideoDetectionPipeline::lastPtsUs,
                   "Presentation timestamp of the last delivered result")
      .def_prop_ro("frame_index", &rcdl::AsyncVideoDetectionPipeline::lastFrameIndex,
                   "Decoder frame index of the last delivered result")
      .def_prop_ro("letterbox", [](const rcdl::AsyncVideoDetectionPipeline& p) {
        return lbToTuple(p.lastLetterbox());
      })
      .def_prop_ro("width", &rcdl::AsyncVideoDetectionPipeline::width)
      .def_prop_ro("height", &rcdl::AsyncVideoDetectionPipeline::height)
      .def_prop_ro("frames_decoded", &rcdl::AsyncVideoDetectionPipeline::framesDecoded)
      .def_prop_ro("using_external_buffers",
                   &rcdl::AsyncVideoDetectionPipeline::usingExternalBuffers)
      .def_prop_ro("workers", &rcdl::AsyncVideoDetectionPipeline::workers)
      .def_prop_ro("head", [](const rcdl::AsyncVideoDetectionPipeline& p) {
        return std::string(rcdl::headName(p.head()));
      })
      .def_prop_ro("profile", [](const rcdl::AsyncVideoDetectionPipeline& p) {
        const rcdl::StageProfile s = p.profile();
        return nb::make_tuple(s.decodePerFrame(), s.preprocPerFrame(), s.inferPerFrame(),
                              s.postprocPerFrame(), s.frames);
      }, "(decode_ms, preproc_ms, infer_ms, postproc_ms, frames) averaged per frame");

  // --- tracking -----------------------------------------------------------------
  nb::class_<rcdl::Track>(m, "Track")
      .def_ro("track_id", &rcdl::Track::track_id)
      .def_ro("x1", &rcdl::Track::x1)
      .def_ro("y1", &rcdl::Track::y1)
      .def_ro("x2", &rcdl::Track::x2)
      .def_ro("y2", &rcdl::Track::y2)
      .def_ro("score", &rcdl::Track::score)
      .def_ro("class_id", &rcdl::Track::class_id)
      .def("__repr__", [](const rcdl::Track& t) {
        char buf[144];
        std::snprintf(buf, sizeof(buf),
                      "Track(id=%d cls=%d score=%.3f box=[%.1f,%.1f,%.1f,%.1f])", t.track_id,
                      t.class_id, t.score, t.x1, t.y1, t.x2, t.y2);
        return std::string(buf);
      });

  nb::class_<rcdl::BoostConfig>(m, "BoostConfig")
      .def(nb::init<>())
      .def_rw("rich_similarity", &rcdl::BoostConfig::rich_similarity)
      .def_rw("soft_biou", &rcdl::BoostConfig::soft_biou)
      .def_rw("boost_detections", &rcdl::BoostConfig::boost_detections)
      .def_rw("lambda_iou", &rcdl::BoostConfig::lambda_iou)
      .def_rw("lambda_mhd", &rcdl::BoostConfig::lambda_mhd)
      .def_rw("lambda_shape", &rcdl::BoostConfig::lambda_shape)
      .def_rw("min_iou", &rcdl::BoostConfig::min_iou)
      .def_rw("dlo_alpha", &rcdl::BoostConfig::dlo_alpha)
      .def_rw("vt_start", &rcdl::BoostConfig::vt_start)
      .def_rw("vt_end", &rcdl::BoostConfig::vt_end)
      .def_rw("vt_steps", &rcdl::BoostConfig::vt_steps)
      .def_rw("duo", &rcdl::BoostConfig::duo)
      .def_rw("duo_iou", &rcdl::BoostConfig::duo_iou);

  nb::class_<rcdl::ByteTrackConfig>(m, "ByteTrackConfig")
      .def(nb::init<>())
      .def_rw("track_thresh", &rcdl::ByteTrackConfig::track_thresh)
      .def_rw("high_thresh", &rcdl::ByteTrackConfig::high_thresh)
      .def_rw("match_thresh", &rcdl::ByteTrackConfig::match_thresh)
      .def_rw("track_buffer", &rcdl::ByteTrackConfig::track_buffer)
      .def_rw("frame_rate", &rcdl::ByteTrackConfig::frame_rate)
      .def_rw("proximity_thresh", &rcdl::ByteTrackConfig::proximity_thresh)
      .def_rw("appearance_thresh", &rcdl::ByteTrackConfig::appearance_thresh)
      .def_rw("ema_alpha", &rcdl::ByteTrackConfig::ema_alpha)
      .def_rw("boost", &rcdl::ByteTrackConfig::boost);

  nb::class_<rcdl::ByteTracker>(m, "ByteTracker")
      .def(nb::init<rcdl::ByteTrackConfig>(), "config"_a = rcdl::ByteTrackConfig())
      .def(
          "update",
          [](rcdl::ByteTracker& t, const std::vector<rcdl::Detection>& dets,
             const std::vector<std::vector<float>>& embeddings) {
            if (embeddings.empty()) return t.update(dets);
            return t.update(dets, embeddings);
          },
          "detections"_a, "embeddings"_a = std::vector<std::vector<float>>(),
          "Advance one frame; returns the active tracks. `embeddings` runs parallel to "
          "`detections` (an empty entry means geometry-only for that detection)")
      .def(
          "apply_camera_motion",
          [](rcdl::ByteTracker& t, const std::vector<float>& affine) {
            if (affine.size() != 6) {
              throw std::invalid_argument("apply_camera_motion: expected a row-major 2x3 matrix");
            }
            t.applyCameraMotion(affine.data());
          },
          "affine"_a, "Warp every tracklet by a previous->current 2x3 affine before update()")
      .def("reset", &rcdl::ByteTracker::reset);

  // TrackingPipeline — detection + ByteTrack (+ optional ReID) in one object.
  // Bound after ByteTracker because it is the composition of it and
  // DetectionPipeline, and because `reid` being optional is the whole point:
  // pass a second Engine and association gains an appearance term.
  nb::class_<rcdl::TrackingPipeline>(m, "TrackingPipeline")
      .def(
          "__init__",
          [](rcdl::TrackingPipeline* self, nb::handle engine_arg, nb::handle reid_arg,
             const std::string& model_input, float conf_thresh, float iou_thresh, int max_dets,
             int num_classes, bool apply_sigmoid, std::uint8_t pad, const std::string& backend,
             const rcdl::ByteTrackConfig& track_cfg, float reid_min_score, int reid_max_crops) {
            rcdl::Engine& engine = engineFrom(engine_arg);
            rcdl::PipelineConfig cfg;
            cfg.model_input = formatFromName(model_input);
            cfg.pad_value = pad;
            cfg.backend = backendFromName(backend);
            cfg.detect.conf_thresh = cfg.ltrb.conf_thresh = conf_thresh;
            cfg.detect.iou_thresh = cfg.ltrb.iou_thresh = iou_thresh;
            cfg.detect.max_dets = cfg.ltrb.max_dets = max_dets;
            cfg.detect.num_classes = cfg.ltrb.num_classes = num_classes;
            cfg.detect.apply_sigmoid = cfg.ltrb.apply_sigmoid = apply_sigmoid;
            if (reid_arg.is_none()) {
              new (self) rcdl::TrackingPipeline(engine, cfg, track_cfg);
              return;
            }
            rcdl::TrackingReidConfig rcfg;
            rcfg.min_score = reid_min_score;
            rcfg.max_crops = reid_max_crops;
            new (self) rcdl::TrackingPipeline(engine, engineFrom(reid_arg), cfg, track_cfg, rcfg);
          },
          "engine"_a, "reid"_a = nb::none(), "model_input"_a = "rgb888", "conf_thresh"_a = 0.25f,
          "iou_thresh"_a = 0.45f, "max_dets"_a = 300, "num_classes"_a = 80,
          "apply_sigmoid"_a = false, "pad"_a = std::uint8_t(114), "backend"_a = "auto",
          "track_config"_a = rcdl::ByteTrackConfig(), "reid_min_score"_a = 0.5f,
          "reid_max_crops"_a = 32,
          // Both Engines must outlive the pipeline: it holds them by reference.
          nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>())
      .def(
          "process",
          [](rcdl::TrackingPipeline& p, const Contig& img, int w, int h, const std::string& fmt,
             int wstride, int hstride) {
            const rcdl::ImageView v = viewFromArray(img, w, h, fmt, wstride, hstride);
            nb::gil_scoped_release nogil;  // preproc + NPU infer + decode + assoc (+ ReID)
            return p.process(v);
          },
          "image"_a, "w"_a, "h"_a, "fmt"_a = "bgr888", "wstride"_a = 0, "hstride"_a = 0,
          "Detect and associate one frame; returns this frame's Tracks in source pixels")
      .def(
          "process_frame",
          [](rcdl::TrackingPipeline& p, rcdl::VideoFrame& f) {
            nb::gil_scoped_release nogil;
            return p.process(f);
          },
          "frame"_a,
          "Track on a decoded VideoFrame. With ReID on, the frame is mapped so the "
          "CPU can read the crops; geometry-only tracking never touches it")
      .def("reset", &rcdl::TrackingPipeline::reset,
           "Drop every tracklet and restart ids at 1 (e.g. on a stream cut)")
      .def_prop_ro("last_detections", &rcdl::TrackingPipeline::lastDetections,
                   "This frame's detections BEFORE association")
      .def_prop_ro("has_reid", &rcdl::TrackingPipeline::hasReid)
      .def_prop_ro("last_embed_count", &rcdl::TrackingPipeline::lastEmbedCount,
                   "Crops embedded on the last frame (0 without ReID) — this is the term "
                   "that makes frame time scale with crowd size")
      .def_prop_ro("letterbox", [](const rcdl::TrackingPipeline& p) {
        return lbToTuple(p.lastLetterbox());
      })
      .def_prop_ro("profile", [](const rcdl::TrackingPipeline& p) {
        const auto& s = p.profile();
        return nb::make_tuple(s.preprocPerFrame(), s.inferPerFrame(), s.postprocPerFrame(),
                              s.frames);
      }, "(preproc_ms, infer_ms, postproc_ms, frames) of the DETECTION half, per frame");

  m.def(
      "normalize_embedding",
      [](const Contig& v) {
        const float* p = floatData(v, "normalize_embedding");
        return rcdl::normalizeEmbedding(p, static_cast<int>(elemCount(v)));
      },
      "vector"_a, "L2-normalize an embedding (raw model output is fine)");

  // tasks/embedding.h declares a second cosineSimilarity(const float*, const
  // float*, int), so the name is overloaded here and has to be disambiguated;
  // the sequence form is the one the Python API has always exposed.
  m.def("cosine_similarity",
        static_cast<float (*)(const std::vector<float>&, const std::vector<float>&)>(
            &rcdl::cosineSimilarity),
        "a"_a, "b"_a);

  m.def(
      "reid_preprocess",
      [](const Contig& img, float x1, float y1, float x2, float y2, const std::string& fmt,
         int out_w, int out_h) {
        if (img.ndim() != 3 || img.shape(2) != 3) {
          throw std::invalid_argument("reid_preprocess: expected an (H, W, 3) uint8 array");
        }
        const int h = static_cast<int>(img.shape(0));
        const int w = static_cast<int>(img.shape(1));
        const rcdl::ImageView v = viewFromArray(img, w, h, fmt, 0, 0);
        // An empty box means "the whole array" — the usual case when the caller
        // has already cropped, and what makes the box arguments optional.
        if (x2 <= x1 || y2 <= y1) {
          x1 = 0.0f;
          y1 = 0.0f;
          x2 = static_cast<float>(w);
          y2 = static_cast<float>(h);
        }
        std::vector<float> out;
        rcdl::reidPreprocess(v, x1, y1, x2, y2, out_w, out_h, rcdl::ReidConfig(), out);
        auto* buf = new float[out.size()];
        std::memcpy(buf, out.data(), out.size() * sizeof(float));
        nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<float*>(p); });
        std::size_t shape[3] = {3, static_cast<std::size_t>(out_h),
                                static_cast<std::size_t>(out_w)};
        return nb::ndarray<nb::numpy, float>(buf, 3, shape, owner);
      },
      "image"_a, "x1"_a = 0.0f, "y1"_a = 0.0f, "x2"_a = 0.0f, "y2"_a = 0.0f,
      "fmt"_a = "bgr888", "out_w"_a = 128, "out_h"_a = 256,
      "Crop and normalize a detection box into a ReID model input (CHW float32)");

  // --- classification -----------------------------------------------------------
  nb::class_<rcdl::ClsConfig>(m, "ClsConfig")
      .def(nb::init<>())
      .def_rw("top_k", &rcdl::ClsConfig::top_k)
      .def_rw("apply_softmax", &rcdl::ClsConfig::apply_softmax)
      .def("__repr__", [](const rcdl::ClsConfig& c) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "ClsConfig(top_k=%d softmax=%d)", c.top_k,
                      static_cast<int>(c.apply_softmax));
        return std::string(buf);
      });

  nb::class_<rcdl::ClsResult>(m, "ClsResult")
      .def_ro("class_id", &rcdl::ClsResult::class_id)
      .def_ro("score", &rcdl::ClsResult::score)
      .def("__repr__", [](const rcdl::ClsResult& r) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "ClsResult(cls=%d score=%.4f)", r.class_id, r.score);
        return std::string(buf);
      });

  nb::class_<rcdl::CropBox>(m, "CropBox")
      .def(nb::init<>())
      .def_rw("x", &rcdl::CropBox::x)
      .def_rw("y", &rcdl::CropBox::y)
      .def_rw("w", &rcdl::CropBox::w)
      .def_rw("h", &rcdl::CropBox::h)
      .def("__repr__", [](const rcdl::CropBox& b) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "CropBox(x=%d y=%d w=%d h=%d)", b.x, b.y, b.w, b.h);
        return std::string(buf);
      });

  m.def(
      "decode_classification",
      [](const Contig& scores, const rcdl::ClsConfig& cfg) -> std::vector<rcdl::ClsResult> {
        const float* p = floatData(scores, "decode_classification");
        if (elemCount(scores) == 0) return {};
        // The array's own shape decides the class count, so a feature map is
        // rejected instead of being read as a very long score vector.
        return rcdl::decodeClassification(p, shapeOf(scores), cfg);
      },
      "scores"_a, "config"_a = rcdl::ClsConfig(),
      "Top-k over a float32 classifier head; returns ClsResults, best first");

  m.def("class_count_from_shape", &rcdl::classCountFromShape, "shape"_a,
        "The single non-unit dimension of a classifier output shape (raises otherwise)");

  m.def("center_crop_box", &rcdl::centerCropBox, "src_w"_a, "src_h"_a, "out_w"_a, "out_h"_a,
        "crop_ratio"_a = 0.875f,
        "Source rectangle of the resize-then-centre-crop eval transform");

  m.def("load_class_labels", &rcdl::loadClassLabels, "path"_a, "strip_wnid"_a = true,
        "One class name per line, in class-index order ('nXXXXXXXX ' prefixes stripped)");

  m.def("class_label", &rcdl::classLabel, "labels"_a, "class_id"_a,
        "labels[class_id], or 'class <id>' when the table is shorter than the model");

  m.def(
      "looks_like_probabilities",
      [](const Contig& values, float tol) {
        const float* p = floatData(values, "looks_like_probabilities");
        return rcdl::looksLikeProbabilities(p, static_cast<int>(elemCount(values)), tol);
      },
      "values"_a, "tol"_a = 1e-2f, "Does this output already have the softmax inside the graph?");

  nb::class_<rcdl::Classifier>(m, "Classifier")
      .def(
          "__init__",
          [](rcdl::Classifier* self, nb::handle engine_arg, int top_k, bool apply_softmax,
             const std::string& model_input, float crop_ratio, const std::string& backend,
             int output_index) {
            rcdl::Engine& engine = engineFrom(engine_arg);
            rcdl::ClsConfig cfg;
            cfg.top_k = top_k;
            cfg.apply_softmax = apply_softmax;
            rcdl::ClsPreproc pre;
            pre.model_input = formatFromName(model_input);
            pre.crop_ratio = crop_ratio;
            pre.backend = backendFromName(backend);
            new (self) rcdl::Classifier(engine, cfg, pre, output_index);
          },
          "engine"_a, "top_k"_a = 5, "apply_softmax"_a = true, "model_input"_a = "rgb888",
          "crop_ratio"_a = 0.875f, "backend"_a = "auto", "output_index"_a = 0,
          nb::keep_alive<1, 2>())
      .def(
          "classify",
          [](rcdl::Classifier& c, const Contig& img, int w, int h, const std::string& fmt,
             int wstride, int hstride) {
            const rcdl::ImageView v = viewFromArray(img, w, h, fmt, wstride, hstride);
            nb::gil_scoped_release nogil;  // RGA crop+resize + NPU infer + decode
            return c.classify(v);
          },
          "image"_a, "w"_a, "h"_a, "fmt"_a = "bgr888", "wstride"_a = 0, "hstride"_a = 0,
          "Centre-crop, resize into the NPU input tensor, infer and decode one frame")
      .def(
          "classify_frame",
          [](rcdl::Classifier& c, const rcdl::VideoFrame& f) {
            nb::gil_scoped_release nogil;  // RGA reads the decoder's dma-buf directly
            return c.classify(f.view());
          },
          "frame"_a, "Classify a decoded VideoFrame without copying it out of the VPU's buffer")
      .def(
          "postprocess",
          [](const rcdl::Classifier& c) {
            nb::gil_scoped_release nogil;
            return c.postprocess();
          },
          "Decode the bound output as it stands (the caller ran preproc + infer)")
      .def("crop_for", &rcdl::Classifier::cropFor, "src_w"_a, "src_h"_a,
           "The source rectangle classify() would crop out of a frame this size")
      .def_prop_ro("num_classes", &rcdl::Classifier::numClasses)
      .def_prop_ro("input_width", &rcdl::Classifier::inputWidth)
      .def_prop_ro("input_height", &rcdl::Classifier::inputHeight)
      .def_prop_ro("backend", [](const rcdl::Classifier& c) {
        return std::string(rcdl::backendName(c.lastBackend()));
      });

  // --- embeddings ---------------------------------------------------------------
  nb::class_<rcdl::EmbedConfig>(m, "EmbedConfig")
      .def(nb::init<>())
      .def_rw("l2_normalize", &rcdl::EmbedConfig::l2_normalize);

  nb::class_<rcdl::EmbedMatch>(m, "EmbedMatch")
      .def_ro("index", &rcdl::EmbedMatch::index)
      .def_ro("score", &rcdl::EmbedMatch::score)
      .def_ro("label", &rcdl::EmbedMatch::label)
      .def("__repr__", [](const rcdl::EmbedMatch& e) {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "EmbedMatch(index=%d score=%.4f label='%s')", e.index,
                      e.score, e.label.c_str());
        return std::string(buf);
      });

  m.def(
      "decode_embedding",
      [](const Contig& tensor, const rcdl::EmbedConfig& cfg) {
        const float* p = floatData(tensor, "decode_embedding");
        const std::vector<float> v = rcdl::decodeEmbedding(p, shapeOf(tensor), cfg);
        return ownedArray<float>(v.data(), {v.size()});
      },
      "tensor"_a, "config"_a = rcdl::EmbedConfig(),
      "One embedding vector out of a float32 model output (L2-normalized by default)");

  m.def("embedding_dim_from_shape", &rcdl::embeddingDimFromShape, "shape"_a,
        "The single non-unit dimension of an embedding output shape (raises otherwise)");

  m.def(
      "cosine_distance",
      [](const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size() || a.empty()) {
          throw std::invalid_argument("cosine_distance: vectors must be non-empty and equal length");
        }
        return rcdl::cosineDistance(a.data(), b.data(), static_cast<int>(a.size()));
      },
      "a"_a, "b"_a, "1 - cosine similarity, the cost a tracker's assignment wants");

  m.def(
      "euclidean_distance",
      [](const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size() || a.empty()) {
          throw std::invalid_argument(
              "euclidean_distance: vectors must be non-empty and equal length");
        }
        return rcdl::euclideanDistance(a.data(), b.data(), static_cast<int>(a.size()));
      },
      "a"_a, "b"_a);

  nb::class_<rcdl::EmbeddingBank>(m, "EmbeddingBank")
      .def(nb::init<>())
      .def("add", &rcdl::EmbeddingBank::add, "vector"_a, "label"_a = "",
           "Append a labelled vector; it is L2-normalized in the bank")
      .def("search", &rcdl::EmbeddingBank::search, "query"_a, "k"_a = 5,
           "Top-k cosine matches, best first")
      .def(
          "row",
          [](const rcdl::EmbeddingBank& b, int i) {
            const float* r = b.row(i);
            if (r == nullptr) throw std::out_of_range("EmbeddingBank.row: index out of range");
            return ownedArray<float>(r, {static_cast<std::size_t>(b.dim())});
          },
          "i"_a, "Entry i as a unit-norm float32 array")
      .def("label", &rcdl::EmbeddingBank::label, "i"_a)
      .def("clear", &rcdl::EmbeddingBank::clear)
      .def("__len__", &rcdl::EmbeddingBank::size)
      .def_prop_ro("dim", &rcdl::EmbeddingBank::dim);

  nb::class_<rcdl::ImageEmbedder>(m, "ImageEmbedder")
      .def(
          "__init__",
          [](rcdl::ImageEmbedder* self, nb::handle engine_arg, bool l2_normalize,
             const std::string& model_input, float box_expand, const std::string& backend,
             int output_index) {
            rcdl::Engine& engine = engineFrom(engine_arg);
            rcdl::EmbedConfig cfg;
            cfg.l2_normalize = l2_normalize;
            rcdl::EmbedPreproc pre;
            pre.model_input = formatFromName(model_input);
            pre.box_expand = box_expand;
            pre.backend = backendFromName(backend);
            new (self) rcdl::ImageEmbedder(engine, cfg, pre, output_index);
          },
          "engine"_a, "l2_normalize"_a = true, "model_input"_a = "rgb888", "box_expand"_a = 1.0f,
          "backend"_a = "auto", "output_index"_a = 0, nb::keep_alive<1, 2>())
      .def(
          "embed",
          [](rcdl::ImageEmbedder& e, const Contig& img, int w, int h, const std::string& fmt,
             float x1, float y1, float x2, float y2, int wstride, int hstride) {
            const rcdl::ImageView v = viewFromArray(img, w, h, fmt, wstride, hstride);
            std::vector<float> out;
            {
              nb::gil_scoped_release nogil;  // RGA crop+resize + NPU infer
              // An empty box means "the whole frame" — the usual case when the
              // caller has already cropped, and what makes the box optional.
              out = (x2 > x1 && y2 > y1) ? e.embed(v, x1, y1, x2, y2) : e.embed(v);
            }
            return ownedArray<float>(out.data(), {out.size()});
          },
          "image"_a, "w"_a, "h"_a, "fmt"_a = "bgr888", "x1"_a = 0.0f, "y1"_a = 0.0f,
          "x2"_a = 0.0f, "y2"_a = 0.0f, "wstride"_a = 0, "hstride"_a = 0,
          "Crop a box (original-image pixels) out of the frame and embed it")
      .def(
          "embed_frame",
          [](rcdl::ImageEmbedder& e, const rcdl::VideoFrame& f, float x1, float y1, float x2,
             float y2) {
            std::vector<float> out;
            {
              nb::gil_scoped_release nogil;  // RGA reads the decoder's dma-buf directly
              out = (x2 > x1 && y2 > y1) ? e.embed(f.view(), x1, y1, x2, y2) : e.embed(f.view());
            }
            return ownedArray<float>(out.data(), {out.size()});
          },
          "frame"_a, "x1"_a = 0.0f, "y1"_a = 0.0f, "x2"_a = 0.0f, "y2"_a = 0.0f,
          "Embed a box of a decoded VideoFrame without copying it out of the VPU's buffer")
      .def(
          "postprocess",
          [](const rcdl::ImageEmbedder& e) {
            std::vector<float> out;
            {
              nb::gil_scoped_release nogil;
              out = e.postprocess();
            }
            return ownedArray<float>(out.data(), {out.size()});
          },
          "Decode the bound output as it stands (the caller ran preproc + infer)")
      .def_prop_ro("dim", &rcdl::ImageEmbedder::dim)
      .def_prop_ro("input_width", &rcdl::ImageEmbedder::inputWidth)
      .def_prop_ro("input_height", &rcdl::ImageEmbedder::inputHeight)
      .def_prop_ro("backend", [](const rcdl::ImageEmbedder& e) {
        return std::string(rcdl::backendName(e.lastBackend()));
      });

  // --- instance segmentation ------------------------------------------------------
  nb::class_<rcdl::InstanceMask>(m, "InstanceMask")
      .def_ro("x1", &rcdl::InstanceMask::x1)
      .def_ro("y1", &rcdl::InstanceMask::y1)
      .def_ro("x2", &rcdl::InstanceMask::x2)
      .def_ro("y2", &rcdl::InstanceMask::y2)
      .def_ro("score", &rcdl::InstanceMask::score)
      .def_ro("class_id", &rcdl::InstanceMask::class_id)
      .def_ro("mask_x0", &rcdl::InstanceMask::mask_x0)
      .def_ro("mask_y0", &rcdl::InstanceMask::mask_y0)
      .def_ro("mask_w", &rcdl::InstanceMask::mask_w)
      .def_ro("mask_h", &rcdl::InstanceMask::mask_h)
      .def_prop_ro(
          "mask",
          [](const rcdl::InstanceMask& im) {
            // (mask_h, mask_w) uint8 0/1, covering [mask_x0, mask_x0+mask_w) x
            // [mask_y0, mask_y0+mask_h) of the SOURCE frame — empty when the
            // decoder ran with compute_masks off.
            return shapedArray<std::uint8_t>(im.mask,
                                             {static_cast<std::size_t>(std::max(0, im.mask_h)),
                                              static_cast<std::size_t>(std::max(0, im.mask_w))});
          },
          // A property's default policy is reference_internal, which nanobind
          // refuses for an ndarray that already owns its buffer through a
          // capsule — and these are copies, so the capsule IS the owner.
          nb::rv_policy::move, "The instance's binary mask as an (h, w) uint8 array")
      .def("at", &rcdl::InstanceMask::at, "x"_a, "y"_a,
           "Mask value at a SOURCE-frame pixel (0 outside the mask window)")
      .def("__repr__", [](const rcdl::InstanceMask& im) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "InstanceMask(cls=%d score=%.3f box=[%.1f,%.1f,%.1f,%.1f] mask=%dx%d@%d,%d)",
                      im.class_id, im.score, im.x1, im.y1, im.x2, im.y2, im.mask_w, im.mask_h,
                      im.mask_x0, im.mask_y0);
        return std::string(buf);
      });

  nb::class_<rcdl::InstanceSegConfig>(m, "InstanceSegConfig")
      .def(nb::init<>())
      .def_rw("num_classes", &rcdl::InstanceSegConfig::num_classes)
      .def_rw("conf_thresh", &rcdl::InstanceSegConfig::conf_thresh)
      .def_rw("iou_thresh", &rcdl::InstanceSegConfig::iou_thresh)
      .def_rw("max_dets", &rcdl::InstanceSegConfig::max_dets)
      .def_rw("strides", &rcdl::InstanceSegConfig::strides)
      .def_rw("reg_max", &rcdl::InstanceSegConfig::reg_max)
      .def_rw("channels_first", &rcdl::InstanceSegConfig::channels_first)
      .def_rw("proto_channels_first", &rcdl::InstanceSegConfig::proto_channels_first)
      .def_rw("apply_sigmoid", &rcdl::InstanceSegConfig::apply_sigmoid)
      .def_rw("mask_thresh", &rcdl::InstanceSegConfig::mask_thresh)
      .def_rw("compute_masks", &rcdl::InstanceSegConfig::compute_masks)
      .def_rw("full_frame_masks", &rcdl::InstanceSegConfig::full_frame_masks);

  m.def(
      "decode_instance_seg",
      [](const std::vector<Contig>& cls, const std::vector<Contig>& box,
         const std::vector<Contig>& mc, const std::vector<std::pair<int, int>>& grid_hw,
         const std::vector<int>& strides, const Contig& proto, const LbTuple& lb, int num_classes,
         float conf_thresh, float iou_thresh, int max_dets, int reg_max, bool channels_first,
         bool proto_channels_first, bool apply_sigmoid, float mask_thresh, bool compute_masks,
         bool full_frame_masks) {
        rcdl::InstanceSegConfig cfg;
        cfg.num_classes = num_classes;
        cfg.conf_thresh = conf_thresh;
        cfg.iou_thresh = iou_thresh;
        cfg.max_dets = max_dets;
        cfg.strides = strides;
        cfg.reg_max = reg_max;
        cfg.channels_first = channels_first;
        cfg.proto_channels_first = proto_channels_first;
        cfg.apply_sigmoid = apply_sigmoid;
        cfg.mask_thresh = mask_thresh;
        cfg.compute_masks = compute_masks;
        cfg.full_frame_masks = full_frame_masks;

        if (proto.ndim() != 3) {
          throw std::invalid_argument(
              "decode_instance_seg: prototype must be 3-D — (C, H, W) with "
              "proto_channels_first, else (H, W, C)");
        }
        const int proto_c = static_cast<int>(proto.shape(proto_channels_first ? 0 : 2));
        const int proto_h = static_cast<int>(proto.shape(proto_channels_first ? 1 : 0));
        const int proto_w = static_cast<int>(proto.shape(proto_channels_first ? 2 : 1));

        if (cls.size() != box.size() || cls.size() != mc.size() || cls.size() != grid_hw.size() ||
            cls.size() != strides.size()) {
          throw std::invalid_argument(
              "decode_instance_seg: cls, box, mc, grid_hw and strides must have the same length");
        }
        // Same reasoning as decode_yolo_ltrb: the decoder indexes with the
        // caller's grid / class / reg_max numbers, and nothing in numpy ties
        // those to the arrays' real sizes.
        const int box_ch = reg_max > 0 ? 4 * reg_max : 4;
        std::vector<const float*> cp, bp, kp;
        cp.reserve(cls.size());
        bp.reserve(box.size());
        kp.reserve(mc.size());
        for (std::size_t i = 0; i < cls.size(); ++i) {
          const std::size_t cells = static_cast<std::size_t>(grid_hw[i].first) *
                                    static_cast<std::size_t>(grid_hw[i].second);
          const std::size_t need_cls = cells * static_cast<std::size_t>(std::max(0, num_classes));
          const std::size_t need_box = cells * static_cast<std::size_t>(box_ch);
          const std::size_t need_mc = cells * static_cast<std::size_t>(std::max(0, proto_c));
          if (elemCount(cls[i]) < need_cls || elemCount(box[i]) < need_box ||
              elemCount(mc[i]) < need_mc) {
            throw std::invalid_argument(
                "decode_instance_seg: scale " + std::to_string(i) + " needs " +
                std::to_string(need_cls) + " cls, " + std::to_string(need_box) + " box and " +
                std::to_string(need_mc) + " mask-coefficient elements for a " +
                std::to_string(grid_hw[i].first) + "x" + std::to_string(grid_hw[i].second) +
                " grid, got " + std::to_string(elemCount(cls[i])) + ", " +
                std::to_string(elemCount(box[i])) + " and " + std::to_string(elemCount(mc[i])));
          }
          cp.push_back(floatData(cls[i], "decode_instance_seg cls"));
          bp.push_back(floatData(box[i], "decode_instance_seg box"));
          kp.push_back(floatData(mc[i], "decode_instance_seg mc"));
        }
        return rcdl::decodeInstanceSeg(cp, bp, kp, grid_hw, num_classes,
                                       floatData(proto, "decode_instance_seg proto"), proto_h,
                                       proto_w, proto_c, cfg, lbFromTuple(lb));
      },
      "cls"_a, "box"_a, "mc"_a, "grid_hw"_a, "strides"_a, "proto"_a, "letterbox"_a,
      "num_classes"_a = 0, "conf_thresh"_a = 0.25f, "iou_thresh"_a = 0.45f, "max_dets"_a = 100,
      "reg_max"_a = 0, "channels_first"_a = true, "proto_channels_first"_a = true,
      "apply_sigmoid"_a = false, "mask_thresh"_a = 0.5f, "compute_masks"_a = true,
      "full_frame_masks"_a = true,
      "Decode a YOLO instance-seg head (per-scale cls/box/mask-coef + prototype) into "
      "InstanceMasks in source pixels");

  using PyInstanceSegmenter = BoundTask<rcdl::InstanceSegmenter>;
  nb::class_<PyInstanceSegmenter>(m, "InstanceSegmenter")
      .def(
          "__init__",
          [](PyInstanceSegmenter* self, nb::handle engine_arg, float conf_thresh, float iou_thresh,
             int max_dets, int num_classes, bool apply_sigmoid, float mask_thresh,
             bool compute_masks, bool full_frame_masks, const std::string& model_input,
             std::uint8_t pad, const std::string& backend) {
            rcdl::Engine& engine = engineFrom(engine_arg);
            // Grids, class/coefficient counts, reg_max, both channel orders and
            // the strides come from the model (resolveInstanceSegHead), so only
            // the thresholds and mask options below are honoured.
            rcdl::InstanceSegConfig cfg;
            cfg.num_classes = num_classes;
            cfg.conf_thresh = conf_thresh;
            cfg.iou_thresh = iou_thresh;
            cfg.max_dets = max_dets;
            cfg.apply_sigmoid = apply_sigmoid;
            cfg.mask_thresh = mask_thresh;
            cfg.compute_masks = compute_masks;
            cfg.full_frame_masks = full_frame_masks;
            new (self) PyInstanceSegmenter(engine, model_input, pad, backend, cfg);
          },
          "engine"_a, "conf_thresh"_a = 0.25f, "iou_thresh"_a = 0.45f, "max_dets"_a = 100,
          "num_classes"_a = 0, "apply_sigmoid"_a = false, "mask_thresh"_a = 0.5f,
          "compute_masks"_a = true, "full_frame_masks"_a = true, "model_input"_a = "rgb888",
          "pad"_a = std::uint8_t(114), "backend"_a = "auto", nb::keep_alive<1, 2>())
      .def(
          "process",
          [](PyInstanceSegmenter& p, const Contig& img, int w, int h, const std::string& fmt,
             int wstride, int hstride) {
            const rcdl::ImageView v = viewFromArray(img, w, h, fmt, wstride, hstride);
            nb::gil_scoped_release nogil;  // RGA letterbox + NPU infer + decode
            p.run(v);
            return p.task.postprocess(p.last_lb);
          },
          "image"_a, "w"_a, "h"_a, "fmt"_a = "bgr888", "wstride"_a = 0, "hstride"_a = 0,
          "Preprocess, infer and decode one frame; masks and boxes in source pixels")
      .def(
          "process_frame",
          [](PyInstanceSegmenter& p, const rcdl::VideoFrame& f) {
            nb::gil_scoped_release nogil;  // RGA reads the decoder's dma-buf directly
            p.run(f.view());
            return p.task.postprocess(p.last_lb);
          },
          "frame"_a, "Segment a decoded VideoFrame without copying it out of the VPU's buffer")
      .def(
          "postprocess",
          [](const PyInstanceSegmenter& p, const LbTuple& lb) {
            nb::gil_scoped_release nogil;
            return p.task.postprocess(lbFromTuple(lb));
          },
          "letterbox"_a, "Decode the bound outputs as they stand, for a given letterbox")
      .def_prop_ro("letterbox",
                   [](const PyInstanceSegmenter& p) { return lbToTuple(p.last_lb); })
      .def_prop_ro("backend",
                   [](const PyInstanceSegmenter& p) {
                     return std::string(rcdl::backendName(p.last_backend));
                   })
      .def_prop_ro("head_layout",
                   [](const PyInstanceSegmenter& p) { return p.task.layout().describe(); });

  // --- semantic segmentation --------------------------------------------------------
  nb::class_<rcdl::SegMask>(m, "SegMask")
      .def(nb::init<>())
      .def_ro("width", &rcdl::SegMask::width)
      .def_ro("height", &rcdl::SegMask::height)
      .def_ro("num_classes", &rcdl::SegMask::num_classes)
      .def_prop_ro(
          "labels",
          [](const rcdl::SegMask& sm) {
            return shapedArray<std::int32_t>(sm.labels,
                                             {static_cast<std::size_t>(std::max(0, sm.height)),
                                              static_cast<std::size_t>(std::max(0, sm.width))});
          },
          nb::rv_policy::move, "Per-pixel class ids as an (h, w) int32 array")
      .def_prop_ro(
          "confidence",
          [](const rcdl::SegMask& sm) -> nb::object {
            if (sm.confidence.empty()) return nb::none();  // score mode was "none"
            return nb::cast(
                shapedArray<float>(sm.confidence,
                                   {static_cast<std::size_t>(std::max(0, sm.height)),
                                    static_cast<std::size_t>(std::max(0, sm.width))}));
          },
          "Per-pixel score as an (h, w) float32 array, or None when it was not computed")
      .def("label_at", &rcdl::SegMask::labelAt, "x"_a, "y"_a)
      .def("__repr__", [](const rcdl::SegMask& sm) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "SegMask(%dx%d, %d classes)", sm.width, sm.height,
                      sm.num_classes);
        return std::string(buf);
      });

  m.def(
      "decode_seg",
      [](const Contig& tensor, bool channels_first, int num_classes, bool argmaxed,
         const std::string& score) {
        rcdl::SegConfig cfg;
        cfg.num_classes = num_classes;
        cfg.channels_first = channels_first;
        cfg.argmaxed = argmaxed;
        cfg.score = segScoreFromName(score);
        return rcdl::decodeSeg(floatData(tensor, "decode_seg"), shapeOf(tensor), cfg);
      },
      "tensor"_a, "channels_first"_a = true, "num_classes"_a = 0, "argmaxed"_a = false,
      "score"_a = "none",
      "Per-pixel argmax of a float32 logit volume ([C,H,W] or [H,W,C]) into a SegMask");

  m.def(
      "seg_resize", [](const rcdl::SegMask& sm, int dst_w, int dst_h) {
        return rcdl::segResize(sm, dst_w, dst_h);
      },
      "mask"_a, "dst_w"_a, "dst_h"_a, "Nearest-neighbour resize of a label map");

  m.def(
      "seg_to_source",
      [](const rcdl::SegMask& sm, const LbTuple& lb) {
        return rcdl::segToSource(sm, lbFromTuple(lb));
      },
      "mask"_a, "letterbox"_a, "Project a label map off the model canvas back onto the frame");

  m.def(
      "seg_colorize",
      [](const rcdl::SegMask& sm) {
        const std::vector<std::uint8_t> bgr = rcdl::segColorize(sm);
        return shapedArray<std::uint8_t>(bgr, {static_cast<std::size_t>(std::max(0, sm.height)),
                                               static_cast<std::size_t>(std::max(0, sm.width)), 3});
      },
      "mask"_a, "Colour a label map with the VOC palette -> (h, w, 3) uint8 BGR");

  m.def(
      "seg_colorize",
      [](const Contig& labels) {
        const rcdl::SegMask sm = segMaskFromArray(labels);
        const std::vector<std::uint8_t> bgr = rcdl::segColorize(sm);
        return shapedArray<std::uint8_t>(bgr, {static_cast<std::size_t>(sm.height),
                                               static_cast<std::size_t>(sm.width), 3});
      },
      "labels"_a, "Colour an (h, w) int32 label array -> (h, w, 3) uint8 BGR");

  m.def("voc_class_name", &rcdl::vocClassName, "class_id"_a);
  m.def("voc_class_names", &rcdl::vocClassNames);

  using PySegmenter = BoundTask<rcdl::Segmenter>;
  nb::class_<PySegmenter>(m, "Segmenter")
      .def(
          "__init__",
          [](PySegmenter* self, nb::handle engine_arg, int num_classes, bool argmaxed,
             const std::string& score, const std::string& model_input, std::uint8_t pad,
             const std::string& backend, int output_index) {
            rcdl::Engine& engine = engineFrom(engine_arg);
            // No channels_first here on purpose: the Segmenter takes the channel
            // order from the output tensor's own fmt, because an NHWC logit
            // volume and an NCHW one are the same bytes in a different order and
            // only the tensor knows which it is.
            rcdl::SegConfig cfg;
            cfg.num_classes = num_classes;
            cfg.argmaxed = argmaxed;
            cfg.score = segScoreFromName(score);
            new (self) PySegmenter(engine, model_input, pad, backend, cfg, output_index);
          },
          "engine"_a, "num_classes"_a = 0, "argmaxed"_a = false, "score"_a = "none",
          "model_input"_a = "rgb888", "pad"_a = std::uint8_t(114), "backend"_a = "auto",
          "output_index"_a = 0, nb::keep_alive<1, 2>())
      .def(
          "process",
          [](PySegmenter& p, const Contig& img, int w, int h, const std::string& fmt, int wstride,
             int hstride) {
            const rcdl::ImageView v = viewFromArray(img, w, h, fmt, wstride, hstride);
            nb::gil_scoped_release nogil;  // RGA letterbox + NPU infer + argmax
            p.run(v);
            // Projected back onto the frame, so the label map lines up with the
            // image the caller passed in rather than with the model canvas.
            return p.task.postprocess(p.last_lb);
          },
          "image"_a, "w"_a, "h"_a, "fmt"_a = "bgr888", "wstride"_a = 0, "hstride"_a = 0,
          "Preprocess, infer and decode one frame; the label map is in source pixels")
      .def(
          "process_frame",
          [](PySegmenter& p, const rcdl::VideoFrame& f) {
            nb::gil_scoped_release nogil;  // RGA reads the decoder's dma-buf directly
            p.run(f.view());
            return p.task.postprocess(p.last_lb);
          },
          "frame"_a, "Segment a decoded VideoFrame without copying it out of the VPU's buffer")
      .def(
          "postprocess",
          [](const PySegmenter& p) {
            nb::gil_scoped_release nogil;
            return p.task.postprocess();
          },
          "Decode the bound output as it stands, at the model's own resolution")
      .def(
          "postprocess",
          [](const PySegmenter& p, const LbTuple& lb) {
            nb::gil_scoped_release nogil;
            return p.task.postprocess(lbFromTuple(lb));
          },
          "letterbox"_a, "Decode the bound output as it stands, projected onto the frame")
      .def_prop_ro("letterbox", [](const PySegmenter& p) { return lbToTuple(p.last_lb); })
      .def_prop_ro("backend", [](const PySegmenter& p) {
        return std::string(rcdl::backendName(p.last_backend));
      });

  // --- monocular depth ----------------------------------------------------------------
  nb::class_<rcdl::DepthConfig>(m, "DepthConfig")
      .def(nb::init<>())
      .def_rw("width", &rcdl::DepthConfig::width)
      .def_rw("height", &rcdl::DepthConfig::height)
      .def_rw("scale", &rcdl::DepthConfig::scale)
      .def_rw("shift", &rcdl::DepthConfig::shift)
      .def_rw("inverse", &rcdl::DepthConfig::inverse)
      .def_rw("inverse_eps", &rcdl::DepthConfig::inverse_eps)
      .def_rw("clip_lo", &rcdl::DepthConfig::clip_lo)
      .def_rw("clip_hi", &rcdl::DepthConfig::clip_hi)
      .def_rw("normalize", &rcdl::DepthConfig::normalize);

  nb::class_<rcdl::DepthMap>(m, "DepthMap")
      .def(nb::init<>())
      .def_ro("width", &rcdl::DepthMap::width)
      .def_ro("height", &rcdl::DepthMap::height)
      .def_ro("vmin", &rcdl::DepthMap::vmin, "smallest value BEFORE normalization")
      .def_ro("vmax", &rcdl::DepthMap::vmax, "largest value BEFORE normalization")
      .def_prop_ro(
          "data",
          [](const rcdl::DepthMap& dm) {
            return shapedArray<float>(dm.data, {static_cast<std::size_t>(std::max(0, dm.height)),
                                                static_cast<std::size_t>(std::max(0, dm.width))});
          },
          nb::rv_policy::move, "The depth map as an (h, w) float32 array")
      .def("at", &rcdl::DepthMap::at, "x"_a, "y"_a)
      .def("__repr__", [](const rcdl::DepthMap& dm) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "DepthMap(%dx%d, range [%.4f, %.4f])", dm.width, dm.height,
                      dm.vmin, dm.vmax);
        return std::string(buf);
      });

  m.def(
      "decode_depth",
      [](const Contig& tensor, int width, int height, float scale, float shift, bool inverse,
         float inverse_eps, float clip_lo, float clip_hi, bool normalize) {
        rcdl::DepthConfig cfg;
        cfg.width = width;
        cfg.height = height;
        cfg.scale = scale;
        cfg.shift = shift;
        cfg.inverse = inverse;
        cfg.inverse_eps = inverse_eps;
        cfg.clip_lo = clip_lo;
        cfg.clip_hi = clip_hi;
        cfg.normalize = normalize;
        return rcdl::decodeDepth(floatData(tensor, "decode_depth"), shapeOf(tensor), cfg);
      },
      "tensor"_a, "width"_a = 0, "height"_a = 0, "scale"_a = 1.0f, "shift"_a = 0.0f,
      "inverse"_a = false, "inverse_eps"_a = 1e-6f, "clip_lo"_a = 0.0f, "clip_hi"_a = 0.0f,
      "normalize"_a = true,
      "Decode a single-channel float32 depth head: affine -> inverse -> clip -> normalize");

  m.def(
      "depth_resize",
      [](const rcdl::DepthMap& dm, int dst_w, int dst_h) {
        return rcdl::depthResize(dm, dst_w, dst_h);
      },
      "map"_a, "dst_w"_a, "dst_h"_a, "Bilinear resize of a depth map");

  m.def(
      "depth_to_source",
      [](const rcdl::DepthMap& dm, const LbTuple& lb) {
        return rcdl::depthToSource(dm, lbFromTuple(lb));
      },
      "map"_a, "letterbox"_a, "Project a depth map off the model canvas back onto the frame");

  m.def(
      "depth_to_gray8",
      [](const rcdl::DepthMap& dm) {
        const std::vector<std::uint8_t> g = rcdl::depthToGray8(dm);
        return shapedArray<std::uint8_t>(g, {static_cast<std::size_t>(std::max(0, dm.height)),
                                             static_cast<std::size_t>(std::max(0, dm.width))});
      },
      "map"_a, "Min-max stretch to an (h, w) uint8 array");

  m.def(
      "depth_to_gray8",
      [](const Contig& data) {
        const rcdl::DepthMap dm = depthMapFromArray(data);
        const std::vector<std::uint8_t> g = rcdl::depthToGray8(dm);
        return shapedArray<std::uint8_t>(g, {static_cast<std::size_t>(dm.height),
                                             static_cast<std::size_t>(dm.width)});
      },
      "data"_a, "Min-max stretch of an (h, w) float32 array to uint8");

  m.def(
      "depth_colorize",
      [](const rcdl::DepthMap& dm) {
        const std::vector<std::uint8_t> bgr = rcdl::depthColorize(dm);
        return shapedArray<std::uint8_t>(bgr, {static_cast<std::size_t>(std::max(0, dm.height)),
                                               static_cast<std::size_t>(std::max(0, dm.width)), 3});
      },
      "map"_a, "Turbo colourmap -> (h, w, 3) uint8 BGR");

  m.def(
      "depth_colorize",
      [](const Contig& data) {
        const rcdl::DepthMap dm = depthMapFromArray(data);
        const std::vector<std::uint8_t> bgr = rcdl::depthColorize(dm);
        return shapedArray<std::uint8_t>(bgr, {static_cast<std::size_t>(dm.height),
                                               static_cast<std::size_t>(dm.width), 3});
      },
      "data"_a, "Turbo colourmap of an (h, w) float32 array -> (h, w, 3) uint8 BGR");

  using PyDepthEstimator = BoundTask<rcdl::DepthEstimator>;
  nb::class_<PyDepthEstimator>(m, "DepthEstimator")
      .def(
          "__init__",
          [](PyDepthEstimator* self, nb::handle engine_arg, float scale, float shift, bool inverse,
             float clip_lo, float clip_hi, bool normalize, const std::string& model_input,
             std::uint8_t pad, const std::string& backend, int output_index) {
            rcdl::Engine& engine = engineFrom(engine_arg);
            rcdl::DepthConfig cfg;
            cfg.scale = scale;
            cfg.shift = shift;
            cfg.inverse = inverse;
            cfg.clip_lo = clip_lo;
            cfg.clip_hi = clip_hi;
            cfg.normalize = normalize;
            new (self) PyDepthEstimator(engine, model_input, pad, backend, cfg, output_index);
          },
          "engine"_a, "scale"_a = 1.0f, "shift"_a = 0.0f, "inverse"_a = false, "clip_lo"_a = 0.0f,
          "clip_hi"_a = 0.0f, "normalize"_a = true, "model_input"_a = "rgb888",
          "pad"_a = std::uint8_t(114), "backend"_a = "auto", "output_index"_a = 0,
          nb::keep_alive<1, 2>())
      .def(
          "process",
          [](PyDepthEstimator& p, const Contig& img, int w, int h, const std::string& fmt,
             int wstride, int hstride) {
            const rcdl::ImageView v = viewFromArray(img, w, h, fmt, wstride, hstride);
            nb::gil_scoped_release nogil;  // RGA letterbox + NPU infer + decode
            p.run(v);
            return p.task.postprocess(p.last_lb);
          },
          "image"_a, "w"_a, "h"_a, "fmt"_a = "bgr888", "wstride"_a = 0, "hstride"_a = 0,
          "Preprocess, infer and decode one frame; the depth map is in source pixels")
      .def(
          "process_frame",
          [](PyDepthEstimator& p, const rcdl::VideoFrame& f) {
            nb::gil_scoped_release nogil;  // RGA reads the decoder's dma-buf directly
            p.run(f.view());
            return p.task.postprocess(p.last_lb);
          },
          "frame"_a, "Depth of a decoded VideoFrame without copying it out of the VPU's buffer")
      .def(
          "postprocess",
          [](const PyDepthEstimator& p) {
            nb::gil_scoped_release nogil;
            return p.task.postprocess();
          },
          "Decode the bound output as it stands, at the model's own resolution")
      .def(
          "postprocess",
          [](const PyDepthEstimator& p, const LbTuple& lb) {
            nb::gil_scoped_release nogil;
            return p.task.postprocess(lbFromTuple(lb));
          },
          "letterbox"_a, "Decode the bound output as it stands, projected onto the frame")
      .def_prop_ro("letterbox", [](const PyDepthEstimator& p) { return lbToTuple(p.last_lb); })
      .def_prop_ro("backend", [](const PyDepthEstimator& p) {
        return std::string(rcdl::backendName(p.last_backend));
      });

  // --- pose ---------------------------------------------------------------------
  nb::class_<rcdl::Keypoint>(m, "Keypoint")
      .def(nb::init<>())
      .def_rw("x", &rcdl::Keypoint::x)
      .def_rw("y", &rcdl::Keypoint::y)
      .def_rw("score", &rcdl::Keypoint::score, "visibility / confidence in [0,1]")
      .def("__repr__", [](const rcdl::Keypoint& k) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "Keypoint(x=%.1f y=%.1f score=%.3f)", k.x, k.y, k.score);
        return std::string(buf);
      });

  nb::class_<rcdl::PoseDetection>(m, "PoseDetection")
      .def(nb::init<>())
      // The person box IS a Detection, not a copy of its four fields, so
      // nms(), the trackers and every drawing helper work on `.box` unchanged.
      .def_ro("box", &rcdl::PoseDetection::box, "The person box, as a plain Detection")
      .def_ro("keypoints", &rcdl::PoseDetection::keypoints,
              "num_keypoints Keypoints in source pixels, in COCO order")
      .def("__repr__", [](const rcdl::PoseDetection& p) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "PoseDetection(score=%.3f box=[%.1f,%.1f,%.1f,%.1f] keypoints=%d)",
                      p.box.score, p.box.x1, p.box.y1, p.box.x2, p.box.y2,
                      static_cast<int>(p.keypoints.size()));
        return std::string(buf);
      });

  nb::class_<rcdl::PoseConfig>(m, "PoseConfig")
      .def(nb::init<>())
      .def_rw("num_classes", &rcdl::PoseConfig::num_classes)
      .def_rw("num_keypoints", &rcdl::PoseConfig::num_keypoints)
      .def_rw("conf_thresh", &rcdl::PoseConfig::conf_thresh)
      .def_rw("iou_thresh", &rcdl::PoseConfig::iou_thresh)
      .def_rw("max_dets", &rcdl::PoseConfig::max_dets)
      .def_rw("strides", &rcdl::PoseConfig::strides)
      .def_rw("reg_max", &rcdl::PoseConfig::reg_max)
      .def_rw("channels_first", &rcdl::PoseConfig::channels_first)
      .def_rw("apply_sigmoid", &rcdl::PoseConfig::apply_sigmoid)
      .def_prop_rw(
          "kpt_decode",
          [](const rcdl::PoseConfig& c) { return std::string(pyKptDecodeName(c.kpt_decode)); },
          [](rcdl::PoseConfig& c, const std::string& n) { c.kpt_decode = kptDecodeFromName(n); },
          "'model_pixels' (the deployed export decodes in-graph) or 'cell_relative' (raw head)")
      .def_rw("kpt_apply_sigmoid", &rcdl::PoseConfig::kpt_apply_sigmoid);

  m.def(
      "decode_pose",
      [](const std::vector<Contig>& cls, const std::vector<Contig>& box,
         const std::vector<Contig>& kpt, const std::vector<std::pair<int, int>>& grid_hw,
         const std::vector<int>& strides, const LbTuple& lb, int num_classes, int num_keypoints,
         float conf_thresh, float iou_thresh, int max_dets, int reg_max, bool channels_first,
         bool apply_sigmoid, const std::string& kpt_decode, bool kpt_apply_sigmoid) {
        rcdl::PoseConfig cfg;
        cfg.num_classes = num_classes;
        cfg.num_keypoints = num_keypoints;
        cfg.conf_thresh = conf_thresh;
        cfg.iou_thresh = iou_thresh;
        cfg.max_dets = max_dets;
        cfg.strides = strides;
        cfg.reg_max = reg_max;
        cfg.channels_first = channels_first;
        cfg.apply_sigmoid = apply_sigmoid;
        cfg.kpt_decode = kptDecodeFromName(kpt_decode);
        cfg.kpt_apply_sigmoid = kpt_apply_sigmoid;
        requireSameLength(cls.size(), box.size(),
                          "decode_pose: cls, box, kpt, grid_hw and strides must have the same "
                          "length");
        requireSameLength(cls.size(), kpt.size(),
                          "decode_pose: cls, box, kpt, grid_hw and strides must have the same "
                          "length");
        requireSameLength(cls.size(), grid_hw.size(),
                          "decode_pose: cls, box, kpt, grid_hw and strides must have the same "
                          "length");
        requireSameLength(cls.size(), strides.size(),
                          "decode_pose: cls, box, kpt, grid_hw and strides must have the same "
                          "length");
        const std::size_t box_ch = reg_max > 0 ? 4u * static_cast<std::size_t>(reg_max) : 4u;
        const std::size_t kpt_ch = 3u * static_cast<std::size_t>(std::max(0, num_keypoints));
        std::vector<const float*> cp, bp, kp;
        cp.reserve(cls.size());
        bp.reserve(box.size());
        kp.reserve(kpt.size());
        for (std::size_t i = 0; i < cls.size(); ++i) {
          const std::size_t cells = static_cast<std::size_t>(grid_hw[i].first) *
                                    static_cast<std::size_t>(grid_hw[i].second);
          requireElems(cls[i], cells * static_cast<std::size_t>(std::max(0, num_classes)),
                       "decode_pose cls", i);
          requireElems(box[i], cells * box_ch, "decode_pose box", i);
          requireElems(kpt[i], cells * kpt_ch, "decode_pose kpt", i);
          cp.push_back(floatData(cls[i], "decode_pose cls"));
          bp.push_back(floatData(box[i], "decode_pose box"));
          kp.push_back(floatData(kpt[i], "decode_pose kpt"));
        }
        return rcdl::decodePose(cp, bp, kp, grid_hw, cfg, lbFromTuple(lb));
      },
      "cls"_a, "box"_a, "kpt"_a, "grid_hw"_a, "strides"_a, "letterbox"_a, "num_classes"_a = 1,
      "num_keypoints"_a = 17, "conf_thresh"_a = 0.25f, "iou_thresh"_a = 0.45f, "max_dets"_a = 300,
      "reg_max"_a = 0, "channels_first"_a = true, "apply_sigmoid"_a = true,
      "kpt_decode"_a = "model_pixels", "kpt_apply_sigmoid"_a = false,
      "Decode the anchor-free LTRB pose head from per-scale float32 cls/box/keypoint buffers. "
      "`kpt[i]` is scale i's own [K*3,H,W] (channels_first) or [H,W,K*3] tensor; the shared "
      "[1,K*3,A] layout of the deployed export is resolved from the model by PoseEstimator.");

  m.def("coco_keypoint_name", &rcdl::cocoKeypointName, "keypoint_id"_a);
  m.def("coco_keypoint_names", &rcdl::cocoKeypointNames);
  m.def("coco_skeleton", &rcdl::cocoSkeleton,
        "COCO-17 bones as 0-based (joint, joint) pairs — the topology a demo draws");

  using PyPoseEstimator = BoundTask<rcdl::PoseEstimator>;
  nb::class_<PyPoseEstimator>(m, "PoseEstimator")
      .def(
          "__init__",
          [](PyPoseEstimator* self, nb::handle engine_arg, float conf_thresh, float iou_thresh,
             int max_dets, int num_classes, int num_keypoints, bool apply_sigmoid,
             const std::string& kpt_decode, bool kpt_apply_sigmoid,
             const std::string& model_input, std::uint8_t pad, const std::string& backend) {
            rcdl::Engine& engine = engineFrom(engine_arg);
            // Grids, keypoint count, reg_max, channel order and strides come
            // from the model (resolvePoseHead), so only the thresholds and the
            // activation conventions below are honoured.
            rcdl::PoseConfig cfg;
            cfg.num_classes = num_classes;
            cfg.num_keypoints = num_keypoints;
            cfg.conf_thresh = conf_thresh;
            cfg.iou_thresh = iou_thresh;
            cfg.max_dets = max_dets;
            cfg.apply_sigmoid = apply_sigmoid;
            cfg.kpt_decode = kptDecodeFromName(kpt_decode);
            cfg.kpt_apply_sigmoid = kpt_apply_sigmoid;
            new (self) PyPoseEstimator(engine, model_input, pad, backend, cfg);
          },
          "engine"_a, "conf_thresh"_a = 0.25f, "iou_thresh"_a = 0.45f, "max_dets"_a = 300,
          "num_classes"_a = 1, "num_keypoints"_a = 17, "apply_sigmoid"_a = true,
          "kpt_decode"_a = "model_pixels", "kpt_apply_sigmoid"_a = false,
          "model_input"_a = "rgb888", "pad"_a = std::uint8_t(114), "backend"_a = "auto",
          nb::keep_alive<1, 2>())
      .def(
          "process",
          [](PyPoseEstimator& p, const Contig& img, int w, int h, const std::string& fmt,
             int wstride, int hstride) {
            const rcdl::ImageView v = viewFromArray(img, w, h, fmt, wstride, hstride);
            nb::gil_scoped_release nogil;  // RGA letterbox + NPU infer + decode
            p.run(v);
            return p.task.postprocess(p.last_lb);
          },
          "image"_a, "w"_a, "h"_a, "fmt"_a = "bgr888", "wstride"_a = 0, "hstride"_a = 0,
          "Preprocess, infer and decode one frame; boxes and joints in source pixels")
      .def(
          "process_frame",
          [](PyPoseEstimator& p, const rcdl::VideoFrame& f) {
            nb::gil_scoped_release nogil;  // RGA reads the decoder's dma-buf directly
            p.run(f.view());
            return p.task.postprocess(p.last_lb);
          },
          "frame"_a, "Pose on a decoded VideoFrame without copying it out of the VPU's buffer")
      .def(
          "postprocess",
          [](const PyPoseEstimator& p, const LbTuple& lb) {
            nb::gil_scoped_release nogil;
            return p.task.postprocess(lbFromTuple(lb));
          },
          "letterbox"_a, "Decode the bound outputs as they stand, for a given letterbox")
      .def_prop_ro("letterbox", [](const PyPoseEstimator& p) { return lbToTuple(p.last_lb); })
      .def_prop_ro("backend",
                   [](const PyPoseEstimator& p) {
                     return std::string(rcdl::backendName(p.last_backend));
                   })
      .def_prop_ro("num_keypoints",
                   [](const PyPoseEstimator& p) { return p.task.layout().num_keypoints; })
      .def_prop_ro("head_layout",
                   [](const PyPoseEstimator& p) { return p.task.layout().describe(); });

  // --- oriented bounding boxes ----------------------------------------------------
  nb::class_<rcdl::RotatedBox>(m, "RotatedBox")
      .def(nb::init<>())
      .def(
          "__init__",
          [](rcdl::RotatedBox* self, float cx, float cy, float w, float h, float angle) {
            new (self) rcdl::RotatedBox{cx, cy, w, h, angle};
          },
          "cx"_a, "cy"_a, "w"_a, "h"_a, "angle"_a)
      .def_rw("cx", &rcdl::RotatedBox::cx)
      .def_rw("cy", &rcdl::RotatedBox::cy)
      .def_rw("w", &rcdl::RotatedBox::w)
      .def_rw("h", &rcdl::RotatedBox::h)
      .def_rw("angle", &rcdl::RotatedBox::angle, "radians, positive = image x toward image y")
      .def_prop_ro(
          "corners",
          [](const rcdl::RotatedBox& r) {
            // x0,y0,..,x3,y3 — consecutive corners share an edge, so the array
            // draws directly as a closed polyline.
            float out[8];
            rcdl::rotatedBoxCorners(r, out);
            return ownedArray<float>(out, {8});
          },
          nb::rv_policy::move, "The 4 corners as a flat (8,) float32 array")
      .def("__repr__", [](const rcdl::RotatedBox& r) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "RotatedBox(c=[%.1f,%.1f] %.1fx%.1f angle=%.3f)", r.cx,
                      r.cy, r.w, r.h, r.angle);
        return std::string(buf);
      });

  nb::class_<rcdl::ObbDetection>(m, "ObbDetection")
      .def(nb::init<>())
      .def_ro("rrect", &rcdl::ObbDetection::rrect)
      .def_ro("score", &rcdl::ObbDetection::score)
      .def_ro("class_id", &rcdl::ObbDetection::class_id)
      .def("__repr__", [](const rcdl::ObbDetection& d) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "ObbDetection(cls=%d score=%.3f c=[%.1f,%.1f] %.1fx%.1f angle=%.3f)",
                      d.class_id, d.score, d.rrect.cx, d.rrect.cy, d.rrect.w, d.rrect.h,
                      d.rrect.angle);
        return std::string(buf);
      });

  nb::class_<rcdl::ObbConfig>(m, "ObbConfig")
      .def(nb::init<>())
      .def_rw("num_classes", &rcdl::ObbConfig::num_classes)
      .def_rw("conf_thresh", &rcdl::ObbConfig::conf_thresh)
      .def_rw("iou_thresh", &rcdl::ObbConfig::iou_thresh)
      .def_rw("max_dets", &rcdl::ObbConfig::max_dets)
      .def_rw("strides", &rcdl::ObbConfig::strides)
      .def_rw("reg_max", &rcdl::ObbConfig::reg_max)
      .def_rw("channels_first", &rcdl::ObbConfig::channels_first)
      .def_rw("apply_sigmoid", &rcdl::ObbConfig::apply_sigmoid)
      .def_rw("apply_angle_sigmoid", &rcdl::ObbConfig::apply_angle_sigmoid)
      .def_rw("angle_bias", &rcdl::ObbConfig::angle_bias)
      .def_rw("regularize", &rcdl::ObbConfig::regularize);

  // Two spellings on purpose: a bound RotatedBox, and the (cx, cy, w, h, angle)
  // tuple a numpy caller already has. Registration order matters — a Python
  // tuple never converts to a RotatedBox, so it falls through to the second.
  m.def(
      "rotated_iou",
      [](const rcdl::RotatedBox& a, const rcdl::RotatedBox& b) { return rcdl::rotatedIoU(a, b); },
      "a"_a, "b"_a, "Intersection-over-union of two rotated rectangles");
  m.def(
      "rotated_iou",
      [](const RrTuple& a, const RrTuple& b) {
        return rcdl::rotatedIoU(rrFromTuple(a), rrFromTuple(b));
      },
      "a"_a, "b"_a,
      "Intersection-over-union of two (cx, cy, w, h, angle) rotated rectangles");

  m.def(
      "rotated_nms",
      [](const Contig& boxes, float iou_thresh, int max_dets) {
        if (boxes.ndim() != 2 || boxes.shape(1) != 7) {
          throw std::invalid_argument(
              "rotated_nms: expected an (N, 7) [cx,cy,w,h,angle,score,class] array");
        }
        const float* p = floatData(boxes, "rotated_nms");
        std::vector<rcdl::ObbDetection> dets(boxes.shape(0));
        for (std::size_t i = 0; i < boxes.shape(0); ++i) {
          const float* r = p + i * 7;
          dets[i] = {{r[0], r[1], r[2], r[3], r[4]}, r[5], static_cast<int>(r[6])};
        }
        return rcdl::rotatedNms(dets, iou_thresh, max_dets);
      },
      "boxes"_a, "iou_thresh"_a = 0.4f, "max_dets"_a = 300,
      "Per-class greedy NMS on rotated IoU over an (N,7) float32 array; returns kept row indices");

  m.def(
      "decode_obb",
      [](const std::vector<Contig>& cls, const std::vector<Contig>& box,
         const std::vector<Contig>& angle, const std::vector<std::pair<int, int>>& grid_hw,
         const std::vector<int>& strides, const LbTuple& lb, int num_classes, float conf_thresh,
         float iou_thresh, int max_dets, int reg_max, bool channels_first, bool apply_sigmoid,
         bool apply_angle_sigmoid, float angle_bias, bool regularize) {
        rcdl::ObbConfig cfg;
        cfg.num_classes = num_classes;
        cfg.conf_thresh = conf_thresh;
        cfg.iou_thresh = iou_thresh;
        cfg.max_dets = max_dets;
        cfg.strides = strides;
        cfg.reg_max = reg_max;
        cfg.channels_first = channels_first;
        cfg.apply_sigmoid = apply_sigmoid;
        cfg.apply_angle_sigmoid = apply_angle_sigmoid;
        cfg.angle_bias = angle_bias;
        cfg.regularize = regularize;
        requireSameLength(cls.size(), box.size(),
                          "decode_obb: cls, box, angle, grid_hw and strides must have the same "
                          "length");
        requireSameLength(cls.size(), angle.size(),
                          "decode_obb: cls, box, angle, grid_hw and strides must have the same "
                          "length");
        requireSameLength(cls.size(), grid_hw.size(),
                          "decode_obb: cls, box, angle, grid_hw and strides must have the same "
                          "length");
        requireSameLength(cls.size(), strides.size(),
                          "decode_obb: cls, box, angle, grid_hw and strides must have the same "
                          "length");
        const std::size_t box_ch = reg_max > 0 ? 4u * static_cast<std::size_t>(reg_max) : 4u;
        std::vector<const float*> cp, bp, ap;
        cp.reserve(cls.size());
        bp.reserve(box.size());
        ap.reserve(angle.size());
        for (std::size_t i = 0; i < cls.size(); ++i) {
          const std::size_t cells = static_cast<std::size_t>(grid_hw[i].first) *
                                    static_cast<std::size_t>(grid_hw[i].second);
          requireElems(cls[i], cells * static_cast<std::size_t>(std::max(0, num_classes)),
                       "decode_obb cls", i);
          requireElems(box[i], cells * box_ch, "decode_obb box", i);
          // One channel is contiguous per cell in every layout, so the angle
          // needs neither a channel order nor a stride — just `cells` values.
          requireElems(angle[i], cells, "decode_obb angle", i);
          cp.push_back(floatData(cls[i], "decode_obb cls"));
          bp.push_back(floatData(box[i], "decode_obb box"));
          ap.push_back(floatData(angle[i], "decode_obb angle"));
        }
        return rcdl::decodeObb(cp, bp, ap, grid_hw, cfg, lbFromTuple(lb));
      },
      "cls"_a, "box"_a, "angle"_a, "grid_hw"_a, "strides"_a, "letterbox"_a, "num_classes"_a = 15,
      "conf_thresh"_a = 0.25f, "iou_thresh"_a = 0.4f, "max_dets"_a = 300, "reg_max"_a = 0,
      "channels_first"_a = true, "apply_sigmoid"_a = true, "apply_angle_sigmoid"_a = false,
      "angle_bias"_a = 0.25f, "regularize"_a = true,
      "Decode the anchor-free LTRB OBB head from per-scale float32 cls/box/angle buffers into "
      "ObbDetections in source pixels");

  m.def("dota_class_name", &rcdl::dotaClassName, "class_id"_a);
  m.def("dota_class_names", &rcdl::dotaClassNames);

  using PyObbDetector = BoundTask<rcdl::ObbDetector>;
  nb::class_<PyObbDetector>(m, "ObbDetector")
      .def(
          "__init__",
          [](PyObbDetector* self, nb::handle engine_arg, float conf_thresh, float iou_thresh,
             int max_dets, int num_classes, bool apply_sigmoid, bool apply_angle_sigmoid,
             float angle_bias, bool regularize, const std::string& model_input, std::uint8_t pad,
             const std::string& backend) {
            rcdl::Engine& engine = engineFrom(engine_arg);
            // Grids, class count, reg_max, channel order and strides come from
            // the model (resolveObbHead); only the thresholds and the angle /
            // regularise conventions below are honoured.
            rcdl::ObbConfig cfg;
            cfg.num_classes = num_classes;
            cfg.conf_thresh = conf_thresh;
            cfg.iou_thresh = iou_thresh;
            cfg.max_dets = max_dets;
            cfg.apply_sigmoid = apply_sigmoid;
            cfg.apply_angle_sigmoid = apply_angle_sigmoid;
            cfg.angle_bias = angle_bias;
            cfg.regularize = regularize;
            new (self) PyObbDetector(engine, model_input, pad, backend, cfg);
          },
          "engine"_a, "conf_thresh"_a = 0.25f, "iou_thresh"_a = 0.4f, "max_dets"_a = 300,
          "num_classes"_a = 15, "apply_sigmoid"_a = true, "apply_angle_sigmoid"_a = false,
          "angle_bias"_a = 0.25f, "regularize"_a = true, "model_input"_a = "rgb888",
          "pad"_a = std::uint8_t(114), "backend"_a = "auto", nb::keep_alive<1, 2>())
      .def(
          "process",
          [](PyObbDetector& p, const Contig& img, int w, int h, const std::string& fmt,
             int wstride, int hstride) {
            const rcdl::ImageView v = viewFromArray(img, w, h, fmt, wstride, hstride);
            nb::gil_scoped_release nogil;  // RGA letterbox + NPU infer + decode
            p.run(v);
            return p.task.postprocess(p.last_lb);
          },
          "image"_a, "w"_a, "h"_a, "fmt"_a = "bgr888", "wstride"_a = 0, "hstride"_a = 0,
          "Preprocess, infer and decode one frame; rotated boxes in source pixels")
      .def(
          "process_frame",
          [](PyObbDetector& p, const rcdl::VideoFrame& f) {
            nb::gil_scoped_release nogil;  // RGA reads the decoder's dma-buf directly
            p.run(f.view());
            return p.task.postprocess(p.last_lb);
          },
          "frame"_a, "Detect on a decoded VideoFrame without copying it out of the VPU's buffer")
      .def(
          "postprocess",
          [](const PyObbDetector& p, const LbTuple& lb) {
            nb::gil_scoped_release nogil;
            return p.task.postprocess(lbFromTuple(lb));
          },
          "letterbox"_a, "Decode the bound outputs as they stand, for a given letterbox")
      .def_prop_ro("letterbox", [](const PyObbDetector& p) { return lbToTuple(p.last_lb); })
      .def_prop_ro("backend",
                   [](const PyObbDetector& p) {
                     return std::string(rcdl::backendName(p.last_backend));
                   })
      .def_prop_ro("num_classes",
                   [](const PyObbDetector& p) { return p.task.layout().num_classes; })
      .def_prop_ro("head_layout",
                   [](const PyObbDetector& p) { return p.task.layout().describe(); });

  // --- OCR: DBNet detection + CRNN/CTC recognition ---------------------------------
  nb::class_<rcdl::TextBox>(m, "TextBox")
      .def(nb::init<>())
      .def_prop_ro(
          "pts",
          [](const rcdl::TextBox& b) {
            // TL, TR, BR, BL as a flat (8,) array: consecutive corners share an
            // edge, so it draws as a closed polyline and warps upright without
            // re-ordering. reshape(4, 2) for corner-wise work.
            return ownedArray<float>(b.pts, {8});
          },
          nb::rv_policy::move, "The quadrilateral's 4 corners as a flat (8,) float32 array")
      .def_ro("x1", &rcdl::TextBox::x1, "axis-aligned bbox of pts")
      .def_ro("y1", &rcdl::TextBox::y1)
      .def_ro("x2", &rcdl::TextBox::x2)
      .def_ro("y2", &rcdl::TextBox::y2)
      .def_ro("score", &rcdl::TextBox::score, "mean probability inside the region, in [0,1]")
      .def("__repr__", [](const rcdl::TextBox& b) {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "TextBox(score=%.3f bbox=[%.1f,%.1f,%.1f,%.1f])", b.score,
                      b.x1, b.y1, b.x2, b.y2);
        return std::string(buf);
      });

  nb::class_<rcdl::OcrDetConfig>(m, "OcrDetConfig")
      .def(nb::init<>())
      .def_rw("bin_thresh", &rcdl::OcrDetConfig::bin_thresh)
      .def_rw("box_thresh", &rcdl::OcrDetConfig::box_thresh)
      .def_rw("unclip_ratio", &rcdl::OcrDetConfig::unclip_ratio)
      .def_rw("min_size", &rcdl::OcrDetConfig::min_size)
      .def_rw("min_box_side", &rcdl::OcrDetConfig::min_box_side)
      .def_rw("max_candidates", &rcdl::OcrDetConfig::max_candidates)
      .def_rw("connectivity", &rcdl::OcrDetConfig::connectivity)
      .def_rw("apply_sigmoid", &rcdl::OcrDetConfig::apply_sigmoid)
      .def("__repr__", [](const rcdl::OcrDetConfig& c) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "OcrDetConfig(bin=%.2f box=%.2f unclip=%.2f min_size=%d)",
                      c.bin_thresh, c.box_thresh, c.unclip_ratio, c.min_size);
        return std::string(buf);
      });

  nb::class_<rcdl::OcrRecConfig>(m, "OcrRecConfig")
      .def(nb::init<>())
      .def_rw("time_major", &rcdl::OcrRecConfig::time_major, "true => [1,T,C], false => [1,C,T]")
      .def_rw("apply_softmax", &rcdl::OcrRecConfig::apply_softmax)
      .def_rw("blank_index", &rcdl::OcrRecConfig::blank_index);

  nb::class_<rcdl::TextLine>(m, "TextLine")
      .def(nb::init<>())
      .def_ro("text", &rcdl::TextLine::text)
      .def_ro("score", &rcdl::TextLine::score,
              "mean confidence over the steps that emitted a character")
      .def("__repr__", [](const rcdl::TextLine& l) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.3f", l.score);
        return "TextLine(text='" + l.text + "' score=" + buf + ")";
      });

  m.def(
      "decode_text_boxes",
      [](const Contig& prob, const rcdl::OcrDetConfig& cfg, const LbTuple& lb) {
        const std::pair<int, int> hw = mapHW(prob, "decode_text_boxes");
        const float* p = floatData(prob, "decode_text_boxes");
        nb::gil_scoped_release nogil;  // connected components + hull fitting
        return rcdl::decodeTextBoxes(p, hw.first, hw.second, cfg, lbFromTuple(lb));
      },
      "prob"_a, "config"_a = rcdl::OcrDetConfig(),
      "letterbox"_a = LbTuple(1.0f, 0.0f, 0.0f, 0, 0, 0, 0),
      "Decode a DB probability map ((h, w) float32) into TextBoxes. The default letterbox has "
      "dst_w == 0, which means 'identity' — coordinates come back in probability-map pixels.");

  m.def(
      "sort_text_boxes",
      [](std::vector<rcdl::TextBox> boxes, float row_tol) {
        rcdl::sortTextBoxes(boxes, row_tol);
        return boxes;
      },
      "boxes"_a, "row_tol"_a = 10.0f,
      "Reading order: top to bottom, then left to right within a visual row. Returns a new list "
      "(the C++ sortTextBoxes() sorts in place; a Python list argument is not mutated).");

  m.def(
      "min_area_quad",
      [](const Contig& points) {
        if (elemCount(points) % 2 != 0) {
          throw std::invalid_argument("min_area_quad: expected an (n, 2) float32 array");
        }
        const float* p = floatData(points, "min_area_quad");
        const int n = static_cast<int>(elemCount(points) / 2);
        float out[8];
        float short_side = 0.0f;
        rcdl::minAreaQuad(p, n, out, &short_side);
        return nb::make_tuple(ownedArray<float>(out, {8}), short_side);
      },
      "points"_a,
      "Minimum-area enclosing rectangle of an (n, 2) point set; returns (corners8, short_side)");

  m.def(
      "unclip_quad",
      [](const Contig& quad, float ratio) {
        float in[8];
        quadFromArray(quad, in, "unclip_quad");
        float out[8];
        rcdl::unclipQuad(in, ratio, out);
        return ownedArray<float>(out, {8});
      },
      "quad"_a, "ratio"_a = 1.5f,
      "Expand a convex quadrilateral outward by the DB unclip distance area*ratio/perimeter");

  m.def(
      "ctc_greedy_decode",
      [](const Contig& logits, const std::vector<std::string>& dict, bool apply_softmax,
         int blank_index) {
        if (logits.ndim() != 2) {
          throw std::invalid_argument(
              "ctc_greedy_decode: expected a row-major (num_steps, num_classes) float32 array");
        }
        rcdl::OcrRecConfig cfg;
        cfg.apply_softmax = apply_softmax;
        cfg.blank_index = blank_index;
        const float* p = floatData(logits, "ctc_greedy_decode");
        return rcdl::ctcGreedyDecode(p, static_cast<int>(logits.shape(0)),
                                     static_cast<int>(logits.shape(1)), dict, cfg);
      },
      "logits"_a, "dict"_a, "apply_softmax"_a = false, "blank_index"_a = 0,
      "CTC best-path decode of a (num_steps, num_classes) float32 head into a TextLine");

  m.def("load_char_dict", &rcdl::loadCharDict, "path"_a, "paddle_special"_a = false,
        "One token per line, class-index order. `paddle_special` prepends the blank and appends "
        "the space token the reference decoder adds to a bare ppocr_keys file.");

  using PyTextDetector = BoundTask<rcdl::TextDetector>;
  nb::class_<PyTextDetector>(m, "TextDetector")
      .def(
          "__init__",
          [](PyTextDetector* self, nb::handle engine_arg, const rcdl::OcrDetConfig& cfg,
             const std::string& model_input, std::uint8_t pad, const std::string& backend,
             int output_index) {
            rcdl::Engine& engine = engineFrom(engine_arg);
            new (self) PyTextDetector(engine, model_input, pad, backend, cfg, output_index);
          },
          "engine"_a, "config"_a = rcdl::OcrDetConfig(), "model_input"_a = "rgb888",
          "pad"_a = std::uint8_t(114), "backend"_a = "auto", "output_index"_a = 0,
          nb::keep_alive<1, 2>())
      .def(
          "process",
          [](PyTextDetector& p, const Contig& img, int w, int h, const std::string& fmt,
             int wstride, int hstride) {
            const rcdl::ImageView v = viewFromArray(img, w, h, fmt, wstride, hstride);
            nb::gil_scoped_release nogil;  // RGA letterbox + NPU infer + decode
            p.run(v);
            return p.task.postprocess(p.last_lb);
          },
          "image"_a, "w"_a, "h"_a, "fmt"_a = "bgr888", "wstride"_a = 0, "hstride"_a = 0,
          "Preprocess, infer and decode one frame; quadrilaterals in source pixels")
      .def(
          "process_frame",
          [](PyTextDetector& p, const rcdl::VideoFrame& f) {
            nb::gil_scoped_release nogil;  // RGA reads the decoder's dma-buf directly
            p.run(f.view());
            return p.task.postprocess(p.last_lb);
          },
          "frame"_a, "Detect text in a decoded VideoFrame without copying it out of the VPU")
      .def(
          "postprocess",
          [](const PyTextDetector& p, const LbTuple& lb) {
            nb::gil_scoped_release nogil;
            return p.task.postprocess(lbFromTuple(lb));
          },
          "letterbox"_a, "Decode the bound output as it stands, for a given letterbox")
      .def_prop_ro("letterbox", [](const PyTextDetector& p) { return lbToTuple(p.last_lb); })
      .def_prop_ro("backend",
                   [](const PyTextDetector& p) {
                     return std::string(rcdl::backendName(p.last_backend));
                   })
      .def_prop_ro("map_width", [](const PyTextDetector& p) { return p.task.mapWidth(); })
      .def_prop_ro("map_height", [](const PyTextDetector& p) { return p.task.mapHeight(); })
      .def("describe", [](const PyTextDetector& p) { return p.task.describe(); });

  using PyTextRecognizer = BoundRecognizer;
  nb::class_<PyTextRecognizer>(m, "TextRecognizer")
      .def(
          "__init__",
          [](PyTextRecognizer* self, nb::handle engine_arg, const std::string& dict_path,
             const rcdl::OcrRecConfig& cfg, bool paddle_special, const std::string& model_input,
             const std::string& fit, float input_scale, float input_shift,
             const std::string& backend, int output_index) {
            rcdl::Engine& engine = engineFrom(engine_arg);
            new (self) PyTextRecognizer(engine, model_input, fit, input_scale, input_shift,
                                        backend, rcdl::loadCharDict(dict_path, paddle_special),
                                        cfg, output_index);
          },
          "engine"_a, "dict_path"_a, "config"_a = rcdl::OcrRecConfig(), "paddle_special"_a = false,
          "model_input"_a = "rgb888", "fit"_a = "stretch", "input_scale"_a = 1.0f / 255.0f,
          "input_shift"_a = 0.0f, "backend"_a = "auto", "output_index"_a = 0,
          nb::keep_alive<1, 2>())
      .def(
          "__init__",
          [](PyTextRecognizer* self, nb::handle engine_arg, const std::vector<std::string>& dict,
             const rcdl::OcrRecConfig& cfg, const std::string& model_input, const std::string& fit,
             float input_scale, float input_shift, const std::string& backend, int output_index) {
            rcdl::Engine& engine = engineFrom(engine_arg);
            // The overload for an already-loaded dictionary: several Engines
            // (one per NPU core) sharing one 6625-entry table.
            new (self) PyTextRecognizer(engine, model_input, fit, input_scale, input_shift,
                                        backend, dict, cfg, output_index);
          },
          "engine"_a, "dict"_a, "config"_a = rcdl::OcrRecConfig(), "model_input"_a = "rgb888",
          "fit"_a = "stretch", "input_scale"_a = 1.0f / 255.0f, "input_shift"_a = 0.0f,
          "backend"_a = "auto", "output_index"_a = 0, nb::keep_alive<1, 2>())
      .def(
          "process",
          [](PyTextRecognizer& p, const Contig& img, int w, int h, const std::string& fmt,
             int wstride, int hstride) {
            const rcdl::ImageView v = viewFromArray(img, w, h, fmt, wstride, hstride);
            nb::gil_scoped_release nogil;  // resize + NPU infer + CTC collapse
            p.run(v);
            return p.task.postprocess();
          },
          "image"_a, "w"_a, "h"_a, "fmt"_a = "bgr888", "wstride"_a = 0, "hstride"_a = 0,
          "Recognize ONE already-cropped, upright text line. The crop-and-warp step between "
          "TextDetector and here is the caller's: a TextBox is a quadrilateral in source pixels, "
          "and warping it upright is a host image operation, not a decode.")
      .def(
          "process_frame",
          [](PyTextRecognizer& p, const rcdl::VideoFrame& f) {
            nb::gil_scoped_release nogil;
            p.run(f.view());
            return p.task.postprocess();
          },
          "frame"_a, "Recognize a decoded VideoFrame that already holds one upright text line")
      .def(
          "postprocess",
          [](const PyTextRecognizer& p) {
            nb::gil_scoped_release nogil;
            return p.task.postprocess();
          },
          "Decode the bound output as it stands (the caller ran preproc + infer)")
      .def_prop_ro("dict", [](const PyTextRecognizer& p) { return p.task.dict(); })
      .def_prop_ro("num_steps", [](const PyTextRecognizer& p) { return p.task.numSteps(); })
      .def_prop_ro("num_classes", [](const PyTextRecognizer& p) { return p.task.numClasses(); })
      .def_prop_ro("time_major", [](const PyTextRecognizer& p) { return p.task.timeMajor(); })
      .def_prop_ro("input_width", [](const PyTextRecognizer& p) { return p.in_w; })
      .def_prop_ro("input_height", [](const PyTextRecognizer& p) { return p.in_h; })
      .def_prop_ro("float_input", [](const PyTextRecognizer& p) { return p.float_input; },
                   "Does the model take f32 pixels (and so `input_scale`/`input_shift`)?")
      .def_prop_ro("backend",
                   [](const PyTextRecognizer& p) {
                     return std::string(rcdl::backendName(p.last_backend));
                   })
      .def("describe", [](const PyTextRecognizer& p) { return p.task.describe(); });

  // --- OCR: text-line orientation ----------------------------------------------
  nb::class_<rcdl::TextOrientation>(m, "TextOrientation")
      .def_ro("label", &rcdl::TextOrientation::label, "0 = upright, 1 = rotated 180 degrees")
      .def_ro("score", &rcdl::TextOrientation::score)
      .def_ro("flip180", &rcdl::TextOrientation::flip180,
              "label == 1 and score > threshold: rotate the crop before recognising it")
      .def("__repr__", [](const rcdl::TextOrientation& o) {
        char buf[80];
        std::snprintf(buf, sizeof(buf), "TextOrientation(label=%d score=%.3f%s)", o.label,
                      o.score, o.flip180 ? " flip180" : "");
        return std::string(buf);
      });

  m.def("ocr_line_fit_width", &rcdl::ocrLineFitWidth, "src_w"_a, "src_h"_a, "dst_w"_a,
        "dst_h"_a,
        "Columns a line crop occupies in a dst_w x dst_h input under PP-OCR's fit "
        "(scale to the height, cap the width, anchor top-left, pad the rest)");

  m.def(
      "decode_text_orientation",
      [](const Contig& scores, float thresh) {
        const float* p = floatData(scores, "decode_text_orientation");
        return rcdl::decodeTextOrientation(p, static_cast<int>(elemCount(scores)), thresh);
      },
      "scores"_a, "thresh"_a = 0.9f,
      "Decode a direction head's output ([1,2]) into a TextOrientation");

  using PyTextAngleClassifier = BoundAngleClassifier;
  nb::class_<PyTextAngleClassifier>(m, "TextAngleClassifier")
      .def(
          "__init__",
          [](PyTextAngleClassifier* self, nb::handle engine_arg, float thresh,
             const std::string& model_input, std::uint8_t pad, const std::string& backend,
             int output_index) {
            rcdl::Engine& engine = engineFrom(engine_arg);
            new (self) PyTextAngleClassifier(engine, model_input, pad, backend, thresh,
                                             output_index);
          },
          "engine"_a, "thresh"_a = 0.9f, "model_input"_a = "bgr888", "pad"_a = std::uint8_t(0),
          "backend"_a = "auto", "output_index"_a = 0, nb::keep_alive<1, 2>())
      .def(
          "process",
          [](PyTextAngleClassifier& p, const Contig& img, int w, int h, const std::string& fmt,
             int wstride, int hstride) {
            const rcdl::ImageView v = viewFromArray(img, w, h, fmt, wstride, hstride);
            nb::gil_scoped_release nogil;  // RGA letterbox -> NPU -> argmax
            p.run(v);
            return p.task.postprocess();
          },
          "image"_a, "w"_a, "h"_a, "fmt"_a = "bgr888", "wstride"_a = 0, "hstride"_a = 0,
          "Is this cropped text line upright or upside down?")
      .def(
          "process_frame",
          [](PyTextAngleClassifier& p, const rcdl::VideoFrame& f) {
            nb::gil_scoped_release nogil;
            p.run(f.view());
            return p.task.postprocess();
          },
          "frame"_a)
      .def(
          "postprocess",
          [](const PyTextAngleClassifier& p) {
            nb::gil_scoped_release nogil;
            return p.task.postprocess();
          },
          "Decode the bound output as it stands (the caller ran preproc + infer)")
      .def_prop_ro("threshold", [](const PyTextAngleClassifier& p) { return p.task.threshold(); })
      .def_prop_ro("input_width", [](const PyTextAngleClassifier& p) { return p.in_w; })
      .def_prop_ro("input_height", [](const PyTextAngleClassifier& p) { return p.in_h; })
      .def_prop_ro("fit_width", [](const PyTextAngleClassifier& p) { return p.last_fit_w; },
                   "Columns the last crop occupied; the rest of the input was padding")
      .def_prop_ro("backend",
                   [](const PyTextAngleClassifier& p) {
                     return std::string(rcdl::backendName(p.last_backend));
                   })
      .def("describe", [](const PyTextAngleClassifier& p) { return p.task.describe(); });

  // --- face detection (RetinaFace) --------------------------------------------------
  nb::class_<rcdl::FaceDetection>(m, "FaceDetection")
      .def(nb::init<>())
      .def_ro("x1", &rcdl::FaceDetection::x1)
      .def_ro("y1", &rcdl::FaceDetection::y1)
      .def_ro("x2", &rcdl::FaceDetection::x2)
      .def_ro("y2", &rcdl::FaceDetection::y2)
      .def_ro("score", &rcdl::FaceDetection::score, "face-class probability in [0,1]")
      .def_ro("landmarks", &rcdl::FaceDetection::landmarks,
              "5 (x, y) pairs in source pixels: left eye, right eye, nose, left mouth, right "
              "mouth ('left' is the viewer's left)")
      .def("__repr__", [](const rcdl::FaceDetection& f) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "FaceDetection(score=%.3f box=[%.1f,%.1f,%.1f,%.1f])",
                      f.score, f.x1, f.y1, f.x2, f.y2);
        return std::string(buf);
      });

  nb::class_<rcdl::FaceConfig>(m, "FaceConfig")
      .def(nb::init<>())
      .def_rw("input_w", &rcdl::FaceConfig::input_w)
      .def_rw("input_h", &rcdl::FaceConfig::input_h)
      .def_rw("conf_thresh", &rcdl::FaceConfig::conf_thresh)
      .def_rw("iou_thresh", &rcdl::FaceConfig::iou_thresh)
      .def_rw("max_faces", &rcdl::FaceConfig::max_faces)
      .def_rw("var_center", &rcdl::FaceConfig::var_center,
              "scales centre AND landmark offsets — never the log-size deltas")
      .def_rw("var_size", &rcdl::FaceConfig::var_size)
      .def_rw("steps", &rcdl::FaceConfig::steps)
      .def_rw("min_sizes", &rcdl::FaceConfig::min_sizes)
      .def_rw("clip", &rcdl::FaceConfig::clip)
      .def_rw("face_class", &rcdl::FaceConfig::face_class)
      .def_rw("apply_softmax", &rcdl::FaceConfig::apply_softmax);

  nb::class_<rcdl::FaceHeadLayout>(m, "FaceHeadLayout")
      .def_ro("loc_index", &rcdl::FaceHeadLayout::loc_index)
      .def_ro("conf_index", &rcdl::FaceHeadLayout::conf_index)
      .def_ro("landm_index", &rcdl::FaceHeadLayout::landm_index)
      .def_ro("num_priors", &rcdl::FaceHeadLayout::num_priors, "N, as read from the model")
      .def_ro("input_w", &rcdl::FaceHeadLayout::input_w)
      .def_ro("input_h", &rcdl::FaceHeadLayout::input_h)
      .def("describe", &rcdl::FaceHeadLayout::describe)
      .def("__repr__", [](const rcdl::FaceHeadLayout& l) { return l.describe(); });

  m.def(
      "generate_priors",
      [](int input_w, int input_h, rcdl::FaceConfig cfg) {
        cfg.input_w = input_w;
        cfg.input_h = input_h;
        return priorsArray(rcdl::generatePriors(cfg));
      },
      "input_w"_a, "input_h"_a, "config"_a = rcdl::FaceConfig(),
      "The prior ('anchor') boxes for a canvas, as an (N, 4) float32 array of normalized "
      "(cx, cy, w, h) in the exact order the network's output rows are in");

  m.def(
      "decode_faces",
      [](const Contig& loc, const Contig& conf, const Contig& landm, const LbTuple& lb,
         const rcdl::FaceConfig& cfg) {
        const float* lp = floatData(loc, "decode_faces loc");
        const float* cp = floatData(conf, "decode_faces conf");
        const float* mp = floatData(landm, "decode_faces landm");
        const std::vector<int> ls = shapeOf(loc), cs = shapeOf(conf), ms = shapeOf(landm);
        nb::gil_scoped_release nogil;
        // The priors are regenerated from cfg here rather than taken as an
        // argument: they are not data, they are the configuration, and
        // decodeFaces() throws when the count does not match the tensors.
        return rcdl::decodeFaces(lp, ls, cp, cs, mp, ms, rcdl::generatePriors(cfg), cfg,
                                 lbFromTuple(lb));
      },
      "loc"_a, "conf"_a, "landm"_a, "letterbox"_a, "config"_a = rcdl::FaceConfig(),
      "Decode the three RetinaFace branches — loc (N,4), conf (N,2), landm (N,10), or their "
      "channels-first transposes — into FaceDetections in source pixels");

  using PyFaceDetector = BoundTask<rcdl::FaceDetector>;
  m.def(
      "arcface_template",
      [](int out_w, int out_h) {
        float tpl[10];
        rcdl::arcFaceTemplate(tpl, out_w, out_h);
        return ownedArray<float>(tpl, {5, 2});
      },
      "out_w"_a = 112, "out_h"_a = 112,
      "The canonical 5-point pose ArcFace-family models are trained on, as (5,2)");

  m.def(
      "similarity_transform",
      [](const Contig& src, const Contig& dst) {
        float s[10], d[10], m6[6];
        quadOrFive(src, s, "similarity_transform src");
        quadOrFive(dst, d, "similarity_transform dst");
        rcdl::similarityTransform(s, d, m6);
        return ownedArray<float>(m6, {2, 3});
      },
      "src"_a, "dst"_a,
      "Least-squares similarity (rotation + uniform scale + translation) mapping five "
      "source points onto five target points, as a (2,3) affine matrix");

  m.def(
      "face_align_transform",
      [](const Contig& landmarks, int out_w, int out_h) {
        float l[10], m6[6];
        quadOrFive(landmarks, l, "face_align_transform landmarks");
        rcdl::faceAlignTransform(l, out_w, out_h, m6);
        return ownedArray<float>(m6, {2, 3});
      },
      "landmarks"_a, "out_w"_a = 112, "out_h"_a = 112,
      "The (2,3) affine that warps a face's five landmarks onto the ArcFace template — "
      "feed it to cv2.warpAffine to get the crop an identity model expects");

  nb::class_<PyFaceDetector>(m, "FaceDetector")
      .def(
          "__init__",
          [](PyFaceDetector* self, nb::handle engine_arg, float conf_thresh, float iou_thresh,
             int max_faces, bool apply_softmax, const rcdl::FaceConfig& base,
             const std::string& model_input, std::uint8_t pad, const std::string& backend) {
            rcdl::Engine& engine = engineFrom(engine_arg);
            // The canvas comes from the model's own input tensor (FaceDetector
            // overrides cfg.input_w/h), and the prior count is cross-checked
            // against the model's anchor count at construction.
            rcdl::FaceConfig cfg = base;
            cfg.conf_thresh = conf_thresh;
            cfg.iou_thresh = iou_thresh;
            cfg.max_faces = max_faces;
            cfg.apply_softmax = apply_softmax;
            new (self) PyFaceDetector(engine, model_input, pad, backend, cfg);
          },
          "engine"_a, "conf_thresh"_a = 0.5f, "iou_thresh"_a = 0.4f, "max_faces"_a = 100,
          "apply_softmax"_a = false, "config"_a = rcdl::FaceConfig(),
          // BGR888, not RGB888 like every YOLO head here: RetinaFace is trained
          // with BGR channel order and BGR mean subtraction. Feeding RGB does
          // not error and does not scatter the landmarks — it quietly costs the
          // model most of its recall (on a two-face frame it finds one).
          "model_input"_a = "bgr888", "pad"_a = std::uint8_t(114), "backend"_a = "auto",
          nb::keep_alive<1, 2>())
      .def(
          "process",
          [](PyFaceDetector& p, const Contig& img, int w, int h, const std::string& fmt,
             int wstride, int hstride) {
            const rcdl::ImageView v = viewFromArray(img, w, h, fmt, wstride, hstride);
            nb::gil_scoped_release nogil;  // RGA letterbox + NPU infer + decode
            p.run(v);
            return p.task.postprocess(p.last_lb);
          },
          "image"_a, "w"_a, "h"_a, "fmt"_a = "bgr888", "wstride"_a = 0, "hstride"_a = 0,
          "Preprocess, infer and decode one frame; boxes and landmarks in source pixels")
      .def(
          "process_frame",
          [](PyFaceDetector& p, const rcdl::VideoFrame& f) {
            nb::gil_scoped_release nogil;  // RGA reads the decoder's dma-buf directly
            p.run(f.view());
            return p.task.postprocess(p.last_lb);
          },
          "frame"_a, "Detect faces in a decoded VideoFrame without copying it out of the VPU")
      .def(
          "postprocess",
          [](const PyFaceDetector& p, const LbTuple& lb) {
            nb::gil_scoped_release nogil;
            return p.task.postprocess(lbFromTuple(lb));
          },
          "letterbox"_a, "Decode the bound outputs as they stand, for a given letterbox")
      .def_prop_ro("letterbox", [](const PyFaceDetector& p) { return lbToTuple(p.last_lb); })
      .def_prop_ro("backend",
                   [](const PyFaceDetector& p) {
                     return std::string(rcdl::backendName(p.last_backend));
                   })
      .def_prop_ro("layout", [](const PyFaceDetector& p) { return p.task.layout(); },
                   nb::rv_policy::copy, "Which output holds which branch, plus the anchor count")
      .def_prop_ro(
          "priors", [](const PyFaceDetector& p) { return priorsArray(p.task.priors()); },
          nb::rv_policy::move, "The generated prior set as an (N, 4) array, in output-row order");
}
