#pragma once

#include "rknn_api.h"

namespace rcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

// ===========================================================================
// CPU implementations of operators this runtime does not have
// ===========================================================================
//
// librknnrt does not implement every ONNX operator, and the two ways it can say
// so are very different:
//
//   * an op it knows but cannot put on the NPU runs on ITS OWN CPU fallback,
//     which is invisible except in the timing;
//   * an op it does not know at all is a hole. The toolkit lowers such a node to
//     a generic CPU node with nothing but a note in the conversion log, and the
//     runtime then reports `Unsupport CPU op: <type>` and **segfaults inside
//     rknn_init** — before any caller has a handle to guard.
//
// GridSample is that second case on librknnrt 2.3.2 (RK3588), which is why
// correlation-based optical flow and any other warping network cannot simply be
// converted and run. The supported route is a CUSTOM OPERATOR: the model is
// built with the node renamed to a `cst`-prefixed type (a conversion-time
// decision, see the recipe in the model zoo), the runtime then loads it without
// complaint, and the implementation is supplied here at run time.
//
// The kernels below are plain C++ over the runtime's own tensor buffers. They
// are not fast — a GridSample lands between two NPU subgraphs, so every call
// costs a round trip of the whole tensor — but they are exact, and they turn "no
// model at all" into "a slow model".

/// Register RCDL's CPU kernels on an Engine's context.
///
/// Must run AFTER rknn_init (the API takes a live context) and BEFORE the first
/// infer(); `Engine` does it during construction unless
/// `EngineOptions::custom_ops` is false. Registering a type the loaded model
/// does not use costs nothing and is not an error, so this is unconditional
/// rather than something a caller has to know to ask for.
///
/// Returns the number of operator types registered, and throws rcdl::Error only
/// if the runtime rejects the registration itself.
int registerCustomOps(Engine& engine);

/// The op type the conversion recipe renames ONNX `GridSample` to. Exposed so a
/// caller can name it in an error message rather than repeating the string.
extern const char* const kGridSampleOpType;

}  // namespace rcdl
