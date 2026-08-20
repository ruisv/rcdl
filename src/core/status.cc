#include "rcdl/core/status.h"

#include <sstream>

#include "rknn_api.h"

namespace rcdl::detail {

const char* rknnErrorName(int code) {
  switch (code) {
    case RKNN_SUCC: return "RKNN_SUCC";
    case RKNN_ERR_FAIL: return "RKNN_ERR_FAIL";
    case RKNN_ERR_TIMEOUT: return "RKNN_ERR_TIMEOUT";
    case RKNN_ERR_DEVICE_UNAVAILABLE: return "RKNN_ERR_DEVICE_UNAVAILABLE";
    case RKNN_ERR_MALLOC_FAIL: return "RKNN_ERR_MALLOC_FAIL";
    case RKNN_ERR_PARAM_INVALID: return "RKNN_ERR_PARAM_INVALID";
    case RKNN_ERR_MODEL_INVALID: return "RKNN_ERR_MODEL_INVALID";
    case RKNN_ERR_CTX_INVALID: return "RKNN_ERR_CTX_INVALID";
    case RKNN_ERR_INPUT_INVALID: return "RKNN_ERR_INPUT_INVALID";
    case RKNN_ERR_OUTPUT_INVALID: return "RKNN_ERR_OUTPUT_INVALID";
    case RKNN_ERR_DEVICE_UNMATCH: return "RKNN_ERR_DEVICE_UNMATCH";
    case RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL: return "RKNN_ERR_INCOMPATIBLE_PRE_COMPILE_MODEL";
    case RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION:
      return "RKNN_ERR_INCOMPATIBLE_OPTIMIZATION_LEVEL_VERSION";
    case RKNN_ERR_TARGET_PLATFORM_UNMATCH: return "RKNN_ERR_TARGET_PLATFORM_UNMATCH";
    default: return "";
  }
}

void check(int ret, const char* expr, const char* file, int line) {
  if (ret != 0) {
    std::ostringstream os;
    os << "RCDL: call failed (code " << ret;
    const char* name = rknnErrorName(ret);
    if (name[0] != '\0') os << " " << name;
    os << "): " << expr << "\n  at " << file << ":" << line;
    throw Error(ret, os.str());
  }
}

void require(bool cond, const char* msg, const char* file, int line) {
  if (!cond) {
    std::ostringstream os;
    os << "RCDL: " << msg << "\n  at " << file << ":" << line;
    throw Error(-1, os.str());
  }
}

}  // namespace rcdl::detail
