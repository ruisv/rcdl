#pragma once

#include <cstddef>
#include <cstdint>

#include "rknn_api.h"

namespace rcdl {

/// Byte size of one element of an RKNN tensor type (0 for unknown).
std::size_t elementSize(rknn_tensor_type type) noexcept;

/// Short dtype name ("f32", "f16", "i8", "u8", ...) matching the numpy-style
/// strings the Python layer uses.
const char* dtypeName(rknn_tensor_type type) noexcept;

/// Convert one fp16 (IEEE binary16) value to fp32.
float halfToFloat(std::uint16_t h) noexcept;

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

}  // namespace rcdl
