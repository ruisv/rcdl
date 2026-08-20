#include "rcdl/backend/output_reader.h"

#include <cmath>
#include <cstring>

#include "rcdl/core/status.h"

namespace rcdl {

std::size_t elementSize(rknn_tensor_type type) noexcept {
  switch (type) {
    case RKNN_TENSOR_FLOAT32: return 4;
    case RKNN_TENSOR_FLOAT16: return 2;
    case RKNN_TENSOR_INT8: return 1;
    case RKNN_TENSOR_UINT8: return 1;
    case RKNN_TENSOR_INT16: return 2;
    case RKNN_TENSOR_UINT16: return 2;
    case RKNN_TENSOR_INT32: return 4;
    case RKNN_TENSOR_UINT32: return 4;
    case RKNN_TENSOR_INT64: return 8;
    case RKNN_TENSOR_BOOL: return 1;
    case RKNN_TENSOR_BFLOAT16: return 2;
    case RKNN_TENSOR_INT4: return 1;  // packed; treated as one byte per element upper bound
    default: return 0;
  }
}

const char* dtypeName(rknn_tensor_type type) noexcept {
  switch (type) {
    case RKNN_TENSOR_FLOAT32: return "f32";
    case RKNN_TENSOR_FLOAT16: return "f16";
    case RKNN_TENSOR_INT8: return "i8";
    case RKNN_TENSOR_UINT8: return "u8";
    case RKNN_TENSOR_INT16: return "i16";
    case RKNN_TENSOR_UINT16: return "u16";
    case RKNN_TENSOR_INT32: return "i32";
    case RKNN_TENSOR_UINT32: return "u32";
    case RKNN_TENSOR_INT64: return "i64";
    case RKNN_TENSOR_BOOL: return "bool";
    case RKNN_TENSOR_BFLOAT16: return "bf16";
    case RKNN_TENSOR_INT4: return "i4";
    default: return "?";
  }
}

float halfToFloat(std::uint16_t h) noexcept {
  // IEEE 754 binary16 -> binary32, handling subnormals / inf / nan.
  std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16;
  std::uint32_t exp = (h >> 10) & 0x1Fu;
  std::uint32_t mant = h & 0x3FFu;
  std::uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      // subnormal: normalise
      int e = -1;
      do {
        ++e;
        mant <<= 1;
      } while ((mant & 0x400u) == 0);
      mant &= 0x3FFu;
      bits = sign | ((127 - 15 - e) << 23) | (mant << 13);
    }
  } else if (exp == 0x1Fu) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, sizeof f);
  return f;
}

void dequantizeToFloat(const rknn_tensor_attr& attr, const void* src, float* dst,
                       std::size_t n) {
  const bool affine = attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC;
  const bool dfp = attr.qnt_type == RKNN_TENSOR_QNT_DFP;
  const float scale = affine ? attr.scale : (dfp ? std::ldexp(1.0f, -attr.fl) : 1.0f);
  const float zp = affine ? static_cast<float>(attr.zp) : 0.0f;
  switch (attr.type) {
    case RKNN_TENSOR_FLOAT32:
      std::memcpy(dst, src, n * sizeof(float));
      return;
    case RKNN_TENSOR_FLOAT16: {
      const auto* p = static_cast<const std::uint16_t*>(src);
      for (std::size_t k = 0; k < n; ++k) dst[k] = halfToFloat(p[k]);
      return;
    }
    case RKNN_TENSOR_INT8: {
      const auto* p = static_cast<const std::int8_t*>(src);
      for (std::size_t k = 0; k < n; ++k) dst[k] = (static_cast<float>(p[k]) - zp) * scale;
      return;
    }
    case RKNN_TENSOR_UINT8: {
      const auto* p = static_cast<const std::uint8_t*>(src);
      for (std::size_t k = 0; k < n; ++k) dst[k] = (static_cast<float>(p[k]) - zp) * scale;
      return;
    }
    case RKNN_TENSOR_INT16: {
      const auto* p = static_cast<const std::int16_t*>(src);
      for (std::size_t k = 0; k < n; ++k) dst[k] = (static_cast<float>(p[k]) - zp) * scale;
      return;
    }
    case RKNN_TENSOR_UINT16: {
      const auto* p = static_cast<const std::uint16_t*>(src);
      for (std::size_t k = 0; k < n; ++k) dst[k] = (static_cast<float>(p[k]) - zp) * scale;
      return;
    }
    case RKNN_TENSOR_INT32: {
      const auto* p = static_cast<const std::int32_t*>(src);
      for (std::size_t k = 0; k < n; ++k) dst[k] = (static_cast<float>(p[k]) - zp) * scale;
      return;
    }
    case RKNN_TENSOR_UINT32: {
      const auto* p = static_cast<const std::uint32_t*>(src);
      for (std::size_t k = 0; k < n; ++k) dst[k] = (static_cast<float>(p[k]) - zp) * scale;
      return;
    }
    case RKNN_TENSOR_INT64: {
      const auto* p = static_cast<const std::int64_t*>(src);
      for (std::size_t k = 0; k < n; ++k) dst[k] = static_cast<float>(p[k]);
      return;
    }
    case RKNN_TENSOR_BOOL: {
      const auto* p = static_cast<const std::uint8_t*>(src);
      for (std::size_t k = 0; k < n; ++k) dst[k] = p[k] ? 1.0f : 0.0f;
      return;
    }
    default:
      throw Error(-1, std::string("RCDL: dequantizeToFloat: unsupported tensor type ") +
                          dtypeName(attr.type));
  }
}

}  // namespace rcdl
