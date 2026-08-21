#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

// ===========================================================================
// Super-resolution by tiling a fixed-size convolutional upscaler
// ===========================================================================
//
// The model takes a FIXED tile (128x128 RGB here) and returns it enlarged by an
// integer factor. An arbitrary image is therefore cut into overlapping tiles,
// upscaled one at a time, and blended back together.
//
// WHY OVERLAP AT ALL: the network is fully convolutional but its receptive
// field is not zero, so pixels near a tile's edge are computed from a truncated
// neighbourhood and come out slightly different from the same pixels computed
// mid-tile. Butt-jointed tiles put that discontinuity in a straight line down
// the image, which the eye finds immediately. Overlapping and cross-fading
// spreads the disagreement over a band instead of concentrating it on a seam.
//
// The blend is a weighted average — every tile contributes `w * pixel` and `w`
// to two accumulators, and the result is their ratio. That is what makes the
// IMAGE BORDER need no special case: a border pixel is covered by exactly one
// tile, so it divides by that tile's own weight and comes back unchanged, even
// though the weight there is small.
//
// TWO THINGS THAT SILENTLY PRODUCE A PLAUSIBLE PICTURE IF YOU GET THEM WRONG:
//
//   * INPUT RANGE. The `/255` lives inside the `.rknn` (the toolkit's
//     `std_values`), so the tensor wants 0..255 — as bytes for a quantized
//     build, and as FLOATS STILL IN 0..255 for a float build. Handing a float
//     build 0..1 gives a dark, low-contrast image that still looks like an
//     upscale; measured against the reference it is 7 dB, where the correct
//     range is 63 dB. `upscale()` writes whichever the Engine asks for.
//   * CHANNEL ORDER. The model is RGB and every image entry point here is BGR,
//     so the tile is swapped on the way in and back on the way out.
//
// A NOTE ON JUDGING THE RESULT: this family is trained perceptually, so it
// scores BELOW a bicubic resize on PSNR against the ground truth (measured:
// 24.6 dB vs 28.4 dB) while looking obviously better — it invents plausible
// texture rather than the blur that minimises squared error. Use PSNR to
// compare a build against the float model it came from, not to decide whether
// the model works. See docs/MODELS.md.

struct SuperResConfig {
  /// Overlap between neighbouring tiles, in INPUT pixels. Also the width of the
  /// cross-fade band (scaled up) — 0 disables blending and butt-joints the
  /// tiles, which is only sensible for checking what the seams would look like.
  int overlap = 16;
};

/// An 8-bit interleaved BGR image owned by value.
struct SrImage {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> data;  ///< height * width * 3, BGR
};

/// Where one tile sits. Origins are in INPUT pixels and always in range, so a
/// tile never hangs off the (possibly padded) source.
struct TilePlacement {
  int x;
  int y;
};

/// Lay out tile origins covering `width` x `height` with `tile_w` x `tile_h`
/// tiles overlapping by `overlap` on both axes.
///
/// The last tile in each direction is pushed flush against the far edge rather
/// than left hanging over it, so the final overlap can be LARGER than requested
/// but never smaller — coverage is exact and no tile reads outside the image.
/// A dimension smaller than one tile yields a single origin at 0; the caller is
/// responsible for having padded the source up to tile size.
std::vector<TilePlacement> planTiles(int width, int height, int tile_w, int tile_h,
                                     int overlap);

/// The cross-fade weight for a pixel at `i` along a `len`-long tile axis, with a
/// `ramp`-pixel fade at each end. Always > 0, which is what lets the blend
/// normalize instead of special-casing the borders.
float tileWeight(int i, int len, int ramp);

/// Engine-bound tiled upscaler.
///
/// The scale factor and tile size are read from the model's own input/output
/// shapes, never configured: a mismatch between what the caller thinks the model
/// does and what it does would silently produce a scrambled mosaic.
class SuperResolver {
 public:
  SuperResolver(Engine& engine, SuperResConfig cfg = {}, int input_index = 0,
                int output_index = 0);

  /// Upscale a whole BGR image, tiling as needed. Returns 8-bit BGR at
  /// `scale()` times the source size. Cost is linear in `lastTileCount()`.
  SrImage upscale(const std::uint8_t* bgr, int width, int height, int stride = 0);

  int scale() const { return scale_; }
  int tile() const { return tile_w_; }
  int tileHeight() const { return tile_h_; }
  /// Number of tiles the last upscale() ran — cost is linear in this.
  int lastTileCount() const { return last_tiles_; }

  const SuperResConfig& config() const { return cfg_; }

 private:
  void writeTile(const std::uint8_t* src, int src_stride, int x, int y);

  Engine& engine_;
  SuperResConfig cfg_;
  int in_idx_, out_idx_;
  int tile_w_ = 0, tile_h_ = 0, scale_ = 0;
  bool float_input_ = false;
  int last_tiles_ = 0;
  std::vector<std::uint8_t> input_u8_;
  std::vector<float> input_f32_;
};

}  // namespace rcdl
