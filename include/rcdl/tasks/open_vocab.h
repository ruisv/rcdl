#pragma once

#include <string>
#include <vector>

namespace rcdl {

// ===========================================================================
// Open-vocabulary detection label table (YOLOE, text-prompt export)
// ===========================================================================
//
// YOLOE is open-vocabulary because its classification branch compares an image
// embedding against a CLIP TEXT embedding of each prompt. That comparison is
// exactly the part that does not belong on an NPU — it needs a 600 MB text
// encoder and a tensor of words. So the text encoder runs ONCE on the conversion
// host and its output is folded into the classification convolution. What
// reaches the board is an ordinary anchor-free head whose class axis happens to
// mean whatever words were used: the decode is UNCHANGED (Detector /
// YoloLtrbDetector / InstanceSegmenter, no new post-processing math).
//
// The vocabulary is therefore a CONVERSION-TIME parameter, not a runtime one:
// different words mean a different `.rknn`. All that survives into the runtime
// is the class_id -> prompt-name mapping, which is what this holds. Keep the
// `.rknn` and its `labels.txt` (one prompt per line) side by side and load them
// together — see requireSize() for why that pairing has to be checked.

/// A class_id -> label-name table for an open-vocabulary detection head.
struct LabelMap {
  std::vector<std::string> names;

  /// Load one label per line (leading/trailing whitespace trimmed, blank lines
  /// dropped). Throws Error(-1) if the file cannot be opened or holds no labels.
  static LabelMap fromFile(const std::string& path);

  /// Build from an in-memory list (e.g. a vocabulary assembled in code).
  static LabelMap fromList(std::vector<std::string> v);

  std::size_t size() const { return names.size(); }
  bool empty() const { return names.empty(); }

  /// Name for a class id, or "?" when out of range — never throws, because
  /// decode may legitimately emit ids beyond a truncated table while debugging.
  const std::string& name(int id) const;

  /// Throw unless this table describes exactly `num_classes` classes.
  ///
  /// Worth a hard check rather than a warning: a labels file left over from a
  /// DIFFERENT vocabulary has no effect on the decode at all — every box still
  /// comes out in the right place with the right score — it only renames the
  /// results. A build with one word removed silently shifts every name by one
  /// from that point on. Read the class count off the model
  /// (YoloHeadLayout::num_classes) and pass it here.
  void requireSize(int num_classes) const;
};

}  // namespace rcdl
