#pragma once

#include <stdexcept>
#include <string>

namespace rcdl {

/// Exception carrying a vendor return code. The RKNN runtime returns int
/// (0 == RKNN_SUCC, negative RKNN_ERR_*); MPP returns MPP_RET (0 == MPP_OK);
/// RGA's im2d returns IM_STATUS (1 == IM_STATUS_SUCCESS, <= 0 failure) and is
/// normalised to 0/non-zero by the wrappers before reaching here.
class Error : public std::runtime_error {
 public:
  Error(int code, const std::string& what) : std::runtime_error(what), code_(code) {}
  int code() const noexcept { return code_; }

 private:
  int code_;
};

namespace detail {
/// Throws rcdl::Error if ret != 0. Used by the RCDL_CHECK macro.
void check(int ret, const char* expr, const char* file, int line);
/// Throws rcdl::Error(-1, msg) if !cond. Used by RCDL_REQUIRE.
void require(bool cond, const char* msg, const char* file, int line);
/// Human-readable name for an RKNN_ERR_* code (falls back to the number).
const char* rknnErrorName(int code);
}  // namespace detail

}  // namespace rcdl

/// Wrap every vendor call that returns 0 on success: RCDL_CHECK(rknn_run(ctx, nullptr));
#define RCDL_CHECK(expr) ::rcdl::detail::check((expr), #expr, __FILE__, __LINE__)
/// Precondition / invariant check with a message: RCDL_REQUIRE(i < n, "index out of range");
#define RCDL_REQUIRE(cond, msg) ::rcdl::detail::require((cond), (msg), __FILE__, __LINE__)
