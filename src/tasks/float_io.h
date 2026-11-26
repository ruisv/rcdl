#pragma once

// Internal: float32 in, float32 out, in the model's own (ONNX) axis order.
// Shared by the speech tasks, whose every input is a computed tensor — mels,
// caches, LSTM state, embeddings, noise — rather than image bytes. Not
// installed.

#include <cstddef>
#include <string>
#include <vector>

namespace rcdl {

class Engine;

namespace detail {

/// Input i's shape in the order the model was exported with. The runtime
/// reports a rank-4 input it lays out channels-last as NHWC; that comes back
/// here as [N, C, H, W], so a caller can compare against the ONNX shape.
std::vector<int> logicalInputShape(const Engine& engine, int i);

/// Output i's shape in the exported order (NCHW for a rank-4 NHWC output).
std::vector<int> logicalOutputShape(const Engine& engine, int i);

/// Index of the input / output with this tensor name, or -1.
int inputIndex(const Engine& engine, const std::string& name);
int outputIndex(const Engine& engine, const std::string& name);

/// Write `n` float32 values, row-major in logicalInputShape(i) order, into
/// input i. A rank-4 input the runtime holds as NHWC is transposed on the way
/// in, through `stage`.
///
/// The input must be presented as FLOAT32: true of every input of a float
/// (fp16) build, and of a quantized input only when the Engine was opened with
/// EngineOptions::float_inputs naming it. An INT32 input (a length, a count)
/// takes the values rounded. A quantized input left on the default
/// image-bytes path throws (`who` prefixes the message): its u8 range has no
/// room for the negative half of a mean-zero tensor, and nothing would report
/// the clipping.
void setInputFloat(Engine& engine, int i, const float* data, std::size_t n,
                   std::vector<float>& stage, const char* who);

/// Output i as float32, row-major in logicalOutputShape(i) order, with `shape`
/// filled in. The pointer is into `scratch` or the runtime's own buffer, and
/// stays valid until the next infer() or until `scratch` changes.
const float* outputFloat(const Engine& engine, int i, std::vector<float>& scratch,
                         std::vector<int>& shape);

}  // namespace detail
}  // namespace rcdl
