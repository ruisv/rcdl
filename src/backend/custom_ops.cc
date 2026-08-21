#include "rcdl/backend/custom_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rcdl/backend/engine.h"
#include "rcdl/backend/output_reader.h"
#include "rcdl/core/status.h"
#include "rknn_custom_op.h"

namespace rcdl {

const char* const kGridSampleOpType = "cstGridSample";

namespace {

// --- element access ----------------------------------------------------------
// A custom op sees the runtime's own buffers, so the element type is whatever
// the graph carries at that point — fp16 for a float build, fp32 if the toolkit
// widened it. Both are handled rather than asserted, because which one arrives
// is a property of the compiled model and not of the operator.

float halfOf(const void* base, std::size_t i) {
  return halfToFloat(static_cast<const std::uint16_t*>(base)[i]);
}

std::uint16_t floatToHalf(float f) {
  // Round-to-nearest-even, with the subnormal and overflow cases the flow
  // tensors actually reach; this is the inverse of halfToFloat().
  std::uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const std::uint32_t sign = (x >> 16) & 0x8000u;
  std::int32_t exp = static_cast<std::int32_t>((x >> 23) & 0xFFu) - 127 + 15;
  std::uint32_t mant = x & 0x7FFFFFu;

  if (((x >> 23) & 0xFFu) == 0xFFu) {  // inf / nan
    return static_cast<std::uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
  }
  if (exp >= 0x1F) return static_cast<std::uint16_t>(sign | 0x7C00u);  // overflow -> inf
  if (exp <= 0) {
    if (exp < -10) return static_cast<std::uint16_t>(sign);            // underflow -> 0
    mant |= 0x800000u;
    const std::uint32_t shift = static_cast<std::uint32_t>(14 - exp);
    const std::uint32_t sub = mant >> shift;
    const std::uint32_t rem = mant & ((1u << shift) - 1);
    const std::uint32_t half = 1u << (shift - 1);
    return static_cast<std::uint16_t>(
        sign | (sub + ((rem > half || (rem == half && (sub & 1))) ? 1 : 0)));
  }
  const std::uint32_t out = (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13);
  const std::uint32_t rem = mant & 0x1FFFu;
  return static_cast<std::uint16_t>(
      sign | (out + ((rem > 0x1000u || (rem == 0x1000u && (out & 1))) ? 1 : 0)));
}

inline float readAt(const void* base, rknn_tensor_type t, std::size_t i) {
  switch (t) {
    case RKNN_TENSOR_FLOAT16: return halfOf(base, i);
    case RKNN_TENSOR_FLOAT32: return static_cast<const float*>(base)[i];
    default: return 0.0f;
  }
}

inline void writeAt(void* base, rknn_tensor_type t, std::size_t i, float v) {
  switch (t) {
    case RKNN_TENSOR_FLOAT16:
      static_cast<std::uint16_t*>(base)[i] = floatToHalf(v);
      break;
    case RKNN_TENSOR_FLOAT32:
      static_cast<float*>(base)[i] = v;
      break;
    default:
      break;
  }
}

bool supportedType(rknn_tensor_type t) {
  return t == RKNN_TENSOR_FLOAT16 || t == RKNN_TENSOR_FLOAT32;
}

/// An integer operator attribute.
///
/// It can arrive as an int OR as text — the runtime hands `mode` back as the
/// quoted string `"bilinear"`, and an int attribute can come the same way. A
/// switch on dtype alone therefore falls through to the default for a value
/// that is present and readable, which for `align_corners` means silently
/// sampling on the other convention: a HALF-PIXEL, position-dependent offset in
/// the correlation lookup. A uniform motion field survives that (every pixel is
/// displaced alike); a rotation does not, and measured that way it is the
/// difference between 0.15 px and 0.75 px of endpoint error.
/// Trim the terminator and the quotes the runtime wraps text attributes in.
std::string unquote(std::string s) {
  while (!s.empty() && (s.back() == '\0' || s.back() == ' ')) s.pop_back();
  if (s.size() >= 2 && s.front() == '"' && s.back() == '"') s = s.substr(1, s.size() - 2);
  return s;
}

int attrInt(rknn_custom_op_context* ctx, const char* name, int fallback) {
  rknn_custom_op_attr a{};
  rknn_custom_op_get_op_attr(ctx, name, &a);
  if (a.n_elems == 0 || a.data == nullptr) return fallback;
  switch (a.dtype) {
    case RKNN_TENSOR_INT64: return static_cast<int>(*static_cast<const std::int64_t*>(a.data));
    case RKNN_TENSOR_INT32: return *static_cast<const std::int32_t*>(a.data);
    case RKNN_TENSOR_INT16: return *static_cast<const std::int16_t*>(a.data);
    case RKNN_TENSOR_INT8: return *static_cast<const std::int8_t*>(a.data);
    case RKNN_TENSOR_UINT8: return *static_cast<const std::uint8_t*>(a.data);
    case RKNN_TENSOR_FLOAT32: return static_cast<int>(*static_cast<const float*>(a.data));
    default: break;
  }
  const std::string text = unquote(std::string(static_cast<const char*>(a.data), a.n_elems));
  if (text.empty()) return fallback;
  char* end = nullptr;
  const long v = std::strtol(text.c_str(), &end, 10);
  return end == text.c_str() ? fallback : static_cast<int>(v);
}

/// A string operator attribute, unwrapped.
///
/// The runtime hands these back QUOTED — `"bilinear"`, not `bilinear` — and
/// with the terminator counted in n_elems, so a straight comparison against the
/// ONNX spelling never matches and the kernel refuses a mode it implements.
std::string attrString(rknn_custom_op_context* ctx, const char* name) {
  rknn_custom_op_attr a{};
  rknn_custom_op_get_op_attr(ctx, name, &a);
  if (a.n_elems == 0 || a.data == nullptr) return std::string();
  return unquote(std::string(static_cast<const char*>(a.data), a.n_elems));
}

/// Say why a kernel refused, once.
///
/// The runtime turns a non-zero return into one line — "fallback cpu failed" —
/// with nothing about which of the preconditions was not met, so the kernel
/// prints that itself. Once, because a graph can call this nine times a frame.
int reject(const char* why) {
  static bool said = false;
  if (!said) {
    said = true;
    std::fprintf(stderr, "RCDL custom op %s: %s\n", kGridSampleOpType, why);
  }
  return RKNN_ERR_PARAM_INVALID;
}

std::string describeTensor(const char* name, const rknn_tensor_attr& a) {
  std::string s = std::string(name) + " [";
  for (std::uint32_t i = 0; i < a.n_dims; ++i) {
    if (i) s += ",";
    s += std::to_string(a.dims[i]);
  }
  s += "] fmt=" + std::to_string(static_cast<int>(a.fmt)) +
       " type=" + std::string(dtypeName(a.type));
  return s;
}

// --- GridSample --------------------------------------------------------------
//
// image [N,C,H,W] sampled at grid [N,Hout,Wout,2] of NORMALIZED coordinates in
// [-1,1], (x, y) in that order — the torch convention, which is x-first while
// every dimension around it is y-first. Bilinear, zero outside.
//
// The layout subtlety is the grid, and the tensor's own `fmt` field does NOT
// settle it: the runtime labels every 4-D tensor NCHW, including one whose last
// dimension is the coordinate pair. So the rule here is the shape — a trailing
// 2 means the pairs are interleaved, exactly as ONNX defines GridSample's grid,
// and a LEADING 2 (with no trailing one) means x and y arrived as two planes.
// Reading the wrong one produces a field that is smooth, plausible and wrong.
// The image needs no such care in the graphs this serves: C is 1, and for C == 1
// the two layouts are the same bytes.
int gridSampleCompute(rknn_custom_op_context* op_ctx, rknn_custom_op_tensor* inputs,
                      std::uint32_t n_inputs, rknn_custom_op_tensor* outputs,
                      std::uint32_t n_outputs) {
  if (n_inputs != 2 || n_outputs != 1) return reject("expected two inputs and one output");

  const rknn_tensor_attr& ia = inputs[0].attr;
  const rknn_tensor_attr& ga = inputs[1].attr;
  const rknn_tensor_attr& oa = outputs[0].attr;
  if (ia.n_dims != 4 || ga.n_dims != 4 || oa.n_dims != 4) {
    return reject(("expected 4-D tensors: " + describeTensor("image", ia) + " / " +
                   describeTensor("grid", ga) + " / " + describeTensor("out", oa))
                      .c_str());
  }
  if (!supportedType(ia.type) || !supportedType(ga.type) || !supportedType(oa.type)) {
    return reject(("only float16/float32 are implemented: " + describeTensor("image", ia) +
                   " / " + describeTensor("grid", ga) + " / " + describeTensor("out", oa))
                      .c_str());
  }

  const bool img_nhwc = ia.fmt == RKNN_TENSOR_NHWC;
  const int n = static_cast<int>(ia.dims[0]);
  const int c = static_cast<int>(img_nhwc ? ia.dims[3] : ia.dims[1]);
  const int h = static_cast<int>(img_nhwc ? ia.dims[1] : ia.dims[2]);
  const int w = static_cast<int>(img_nhwc ? ia.dims[2] : ia.dims[3]);

  const bool grid_interleaved = ga.dims[3] == 2;
  const int oh = static_cast<int>(grid_interleaved ? ga.dims[1] : ga.dims[2]);
  const int ow = static_cast<int>(grid_interleaved ? ga.dims[2] : ga.dims[3]);
  if ((!grid_interleaved && ga.dims[1] != 2) || static_cast<int>(ga.dims[0]) != n) {
    return reject(("the grid must be [N,Hout,Wout,2] (or [N,2,Hout,Wout]) matching the "
                   "image's N: " + describeTensor("image", ia) + " / " +
                   describeTensor("grid", ga))
                      .c_str());
  }

  const int align = attrInt(op_ctx, "align_corners", 0);
  const std::string mode = attrString(op_ctx, "mode");
  const std::string pad = attrString(op_ctx, "padding_mode");
  // An empty string means the attribute was not carried through; the ONNX
  // defaults are exactly what is implemented here, so that case is fine. A
  // NAMED mode this kernel does not implement is not.
  if (!mode.empty() && mode.compare(0, 8, "bilinear") != 0 && mode != "linear") {
    return reject(("only bilinear is implemented, model asks for " + mode).c_str());
  }
  if (!pad.empty() && pad.compare(0, 4, "zero") != 0) {
    return reject(("only zero padding is implemented, model asks for " + pad).c_str());
  }

  // One line per call when asked: the shapes, types and buffer offsets the
  // runtime actually hands over are the whole contract, and none of them is
  // documented.
  if (std::getenv("RCDL_CUSTOM_OP_VERBOSE") != nullptr) {
    std::fprintf(stderr,
                 "%s: %s off=%zu size=%u | %s off=%zu size=%u | %s off=%zu size=%u "
                 "| align=%d mode=%s pad=%s\n",
                 kGridSampleOpType, describeTensor("image", ia).c_str(),
                 static_cast<std::size_t>(inputs[0].mem.offset), inputs[0].mem.size,
                 describeTensor("grid", ga).c_str(),
                 static_cast<std::size_t>(inputs[1].mem.offset), inputs[1].mem.size,
                 describeTensor("out", oa).c_str(),
                 static_cast<std::size_t>(outputs[0].mem.offset), outputs[0].mem.size,
                 attrInt(op_ctx, "align_corners", -1), attrString(op_ctx, "mode").c_str(),
                 attrString(op_ctx, "padding_mode").c_str());
  }

  const void* img = inputs[0].mem.virt_addr;
  const void* grid = inputs[1].mem.virt_addr;
  void* out = outputs[0].mem.virt_addr;
  if (!img || !grid || !out) return reject("the runtime handed over a null buffer");

  const std::size_t img_plane = static_cast<std::size_t>(h) * w;
  const std::size_t grid_plane = static_cast<std::size_t>(oh) * ow;
  const std::size_t out_plane = grid_plane;

#pragma omp parallel for schedule(static)
  for (int b = 0; b < n; ++b) {
    const std::size_t gbase = static_cast<std::size_t>(b) * grid_plane * 2;
    const std::size_t ibase = static_cast<std::size_t>(b) * img_plane * c;
    const std::size_t obase = static_cast<std::size_t>(b) * out_plane * c;
    for (std::size_t p = 0; p < grid_plane; ++p) {
      const float gx = grid_interleaved ? readAt(grid, ga.type, gbase + p * 2)
                                        : readAt(grid, ga.type, gbase + p);
      const float gy = grid_interleaved ? readAt(grid, ga.type, gbase + p * 2 + 1)
                                        : readAt(grid, ga.type, gbase + grid_plane + p);
      const float x = align ? (gx + 1.0f) * (w - 1) * 0.5f : ((gx + 1.0f) * w - 1.0f) * 0.5f;
      const float y = align ? (gy + 1.0f) * (h - 1) * 0.5f : ((gy + 1.0f) * h - 1.0f) * 0.5f;

      const int x0 = static_cast<int>(std::floor(x));
      const int y0 = static_cast<int>(std::floor(y));
      const float fx = x - x0, fy = y - y0;
      const float wts[4] = {(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy};
      const int xs[4] = {x0, x0 + 1, x0, x0 + 1};
      const int ys[4] = {y0, y0, y0 + 1, y0 + 1};

      for (int ch = 0; ch < c; ++ch) {
        float acc = 0.0f;
        for (int k = 0; k < 4; ++k) {
          if (xs[k] < 0 || xs[k] >= w || ys[k] < 0 || ys[k] >= h || wts[k] == 0.0f) continue;
          const std::size_t o = img_nhwc
                                    ? ibase + (static_cast<std::size_t>(ys[k]) * w + xs[k]) * c + ch
                                    : ibase + static_cast<std::size_t>(ch) * img_plane +
                                          static_cast<std::size_t>(ys[k]) * w + xs[k];
          acc += wts[k] * readAt(img, ia.type, o);
        }
        const std::size_t d = (oa.fmt == RKNN_TENSOR_NHWC)
                                  ? obase + p * c + ch
                                  : obase + static_cast<std::size_t>(ch) * out_plane + p;
        writeAt(out, oa.type, d, acc);
      }
    }
  }
  return 0;
}

}  // namespace

int registerCustomOps(Engine& engine) {
  rknn_custom_op op;
  std::memset(&op, 0, sizeof(op));
  op.version = 1;
  op.target = RKNN_TARGET_TYPE_CPU;
  std::snprintf(op.op_type, sizeof(op.op_type), "%s", kGridSampleOpType);
  op.compute = gridSampleCompute;

  RCDL_CHECK(rknn_register_custom_ops(engine.handle(), &op, 1));
  return 1;
}

}  // namespace rcdl
