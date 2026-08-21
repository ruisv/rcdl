#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "rknn_api.h"

namespace rcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

/// Byte size of one element of an RKNN tensor type (0 for unknown).
std::size_t elementSize(rknn_tensor_type type) noexcept;

/// Short dtype name ("f32", "f16", "i8", "u8", ...) matching the numpy-style
/// strings the Python layer uses.
const char* dtypeName(rknn_tensor_type type) noexcept;

/// Convert one fp16 (IEEE binary16) value to fp32.
float halfToFloat(std::uint16_t h) noexcept;

/// Convert one fp32 value to fp16 (IEEE binary16), round-to-nearest-even.
///
/// The inverse of halfToFloat(), and the direction a CPU kernel needs when it
/// has to hand results back to a graph that carries fp16 between its stages
/// (backend/custom_ops.h). Subnormals and overflow are handled rather than
/// clamped: a flow field that saturates to inf would be worse than one that
/// loses a bit of precision.
std::uint16_t floatToHalf(float f) noexcept;

/// Dequantize / convert `n` elements of an output tensor into float32.
///
/// Handles the RKNN output encodings:
///   - INT8 / UINT8 with RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC:  (q - zp) * scale
///   - INT8 / INT16 with RKNN_TENSOR_QNT_DFP:                q / 2^fl
///   - FLOAT16 -> FLOAT32 conversion
///   - FLOAT32 passthrough; INT32 / INT64 / BOOL cast
/// `src` must hold `n` packed elements of `attr.type`. Throws for unsupported types.
void dequantizeToFloat(const rknn_tensor_attr& attr, const void* src, float* dst,
                       std::size_t n);

/// Numerically-stable logistic sigmoid. Shared by every YOLO-family decoder
/// (class scores, keypoint / angle activations are all sigmoid logits).
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

/// Inverse sigmoid — turns a probability threshold into a logit threshold so a
/// decoder can compare raw int8/float logits without calling exp() per element.
inline float logit(float p) { return std::log(p / (1.0f - p)); }

/// Return a logical row-major float view of Engine output[out_idx], filling
/// `shape` with its logical shape.
///
/// FAST PATH: an already-FLOAT32 tensor with packed (unpadded) rows is returned
/// ZERO-COPY as a direct pointer into the runtime's output buffer — no dequant
/// walk, no allocation. Detection / pose / seg heads emit 10^5–10^6 elements per
/// frame, so skipping the gather is the difference between ~10 ms and ~1 ms of
/// post-processing.
///
/// SLOW PATH: any other encoding (the usual int8-affine RKNN output, fp16, or a
/// stride-padded row layout) is gathered + dequantized into `scratch` and that
/// pointer is returned. The returned pointer stays valid until `scratch` is
/// destroyed or the next infer().
const float* outputAsFloat(const Engine& engine, int out_idx, std::vector<float>& scratch,
                           std::vector<int>& shape);

/// Affine-quantization parameters of an output tensor, for decoders that
/// threshold in the quantized domain instead of dequantizing everything.
/// `is_affine` is false when the tensor is not INT8/UINT8 affine-quantized.
struct QuantParams {
  bool is_affine = false;
  float scale = 1.0f;
  std::int32_t zero_point = 0;
};
QuantParams quantParams(const rknn_tensor_attr& attr) noexcept;

}  // namespace rcdl
