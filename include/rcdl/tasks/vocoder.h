#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rcdl/preproc/istft.h"

namespace rcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

// ===========================================================================
// Speech synthesis back end: Vocos vocoder (mel spectrogram -> waveform)
// ===========================================================================
//
// Vocos predicts a magnitude and a phase per STFT bin and leaves the waveform
// to an inverse STFT, so the network is a short stack of 1-D convolutions with
// no upsampling — the NPU runs it, the CPU runs the iSTFT.
//
// The compiled graph stops at the head's linear output: log-magnitude and raw
// phase. exp, the magnitude clip, cos and sin run on the CPU. Raw phase spans
// about +/-42 rad, and cos/sin of it is the step reduced precision cannot
// take: a quantizing compiler lowers it to a quantised range reduction, and
// the phase came out at cosine 0.58 against float in an int8 build. As a
// plain linear output the phase survives; fp16 carries +/-42 rad to ~0.03.
//
// The compiled model takes a fixed number of mel frames. Longer input is cut
// into chunks that overlap by `context` frames on each side and only each
// chunk's middle is kept; since every frame depends on at most 27 neighbours
// either way (nine kernel-7 convolutions), a context of >= 27 makes the
// chunk seams exact. The only approximation is at the two ends of the
// utterance, where frames past the edge are filled with `pad_value` instead of
// the network's own zero padding.

struct VocosConfig {
  IstftConfig stft;        ///< n_fft 1024, hop 256, centred — the published Vocos models
  int context = 32;        ///< overlap per side, in mel frames; >= 27 for exact seams
  /// Log-mel value for frames past either end. Silence in these mels sits near
  /// -22; -16 measured closest to whole-utterance Vocos at the edges.
  float pad_value = -16.0f;
  float max_magnitude = 100.0f;  ///< exp(log_mag) is clipped here, as in Vocos
  int sample_rate = 16000; ///< informational: the model's rate (not read from it)
};

/// Engine-bound Vocos: in [1, mels, T] -> out log_mag, phase, each [1, n_fft/2+1, T].
/// Outputs are bound by name (`log_mag`, `phase`), else by position.
class VocosVocoder {
 public:
  explicit VocosVocoder(Engine& engine, VocosConfig cfg = {});

  /// `mel` is [numMels(), frames] row-major (the layout Matcha-TTS emits).
  /// Fills `pcm` with (frames - 1) * hop samples and returns that count.
  int synthesize(const float* mel, int frames, std::vector<float>& pcm);

  int numMels() const noexcept { return mels_; }
  int chunkFrames() const noexcept { return chunk_; }               ///< compiled T
  int bodyFrames() const noexcept { return chunk_ - 2 * cfg_.context; }  ///< kept per chunk
  int sampleRate() const noexcept { return cfg_.sample_rate; }
  const VocosConfig& config() const noexcept { return cfg_; }

 private:
  Engine& eng_;
  VocosConfig cfg_;
  Istft istft_;
  int mels_ = 0, chunk_ = 0, bins_ = 0, out_mag_ = 0, out_phase_ = 1;
  std::vector<float> in_, mag_, cos_, sin_;  // [bins, frames] each
  std::vector<float> stage_;
};

}  // namespace rcdl
