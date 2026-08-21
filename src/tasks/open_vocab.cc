#include "rcdl/tasks/open_vocab.h"

#include <fstream>
#include <string>
#include <utility>

#include "rcdl/core/status.h"

namespace rcdl {

LabelMap LabelMap::fromFile(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw Error(-1, "RCDL LabelMap: cannot open labels file: " + path);
  LabelMap m;
  std::string line;
  while (std::getline(f, line)) {
    // Trim CR (Windows line endings) plus surrounding whitespace.
    const char* ws = " \t\r\n";
    const std::size_t b = line.find_first_not_of(ws);
    if (b == std::string::npos) continue;  // blank line
    const std::size_t e = line.find_last_not_of(ws);
    m.names.push_back(line.substr(b, e - b + 1));
  }
  if (m.names.empty()) throw Error(-1, "RCDL LabelMap: no labels in " + path);
  return m;
}

LabelMap LabelMap::fromList(std::vector<std::string> v) {
  LabelMap m;
  m.names = std::move(v);
  return m;
}

const std::string& LabelMap::name(int id) const {
  static const std::string kUnknown = "?";
  if (id < 0 || id >= static_cast<int>(names.size())) return kUnknown;
  return names[static_cast<std::size_t>(id)];
}

void LabelMap::requireSize(int num_classes) const {
  if (static_cast<int>(names.size()) != num_classes) {
    throw Error(-1, "RCDL LabelMap: the model has " + std::to_string(num_classes) +
                        " classes but this table names " + std::to_string(names.size()) +
                        " — a labels file from a different vocabulary renames every "
                        "detection without changing a single box");
  }
}

}  // namespace rcdl
