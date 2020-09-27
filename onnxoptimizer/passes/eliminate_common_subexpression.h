// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateCommonSubexpression final : public PredicateBasedPass {
  explicit EliminateCommonSubexpression()
      : PredicateBasedPass(
            PassType::Nop,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_common_subexpression";
  }

  bool patternMatchPredicate(Node* node) override {
    // We only check if all the input values are used more than once.
    if (node->inputs().size() == 0) {
        // If we have no inputs it's difficult to find another node without a whole pass optimization.
        return false;
    }
    for (auto i: node->inputs()) {
      if (i->uses().size() == 1) {
        return false;
      }
    }
    return true;
  }

  bool runTransform(Node* node, Graph&, NodeDestroyType& destroy_current)
      override {
    auto node_type = node->kind();
    return false;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE