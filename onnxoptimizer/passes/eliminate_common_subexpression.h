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

  bool have_equal_attributes(Node* source, Node* target) {
    if (!source->hasAttributes() && !target->hasAttributes()) {
      return true;
    }
    else if (source->hasAttributes() != target->hasAttributes()) {
      return false;
    }
    else if (source->attributeNames().size() != target->attributeNames().size()) {
      return false;
    }
    else {
      auto attributeNames = source->attributeNames();

      for (auto name : attributeNames) {
        if (!target->hasAttribute(name)) {
          return false;
        }
        else if (source->kindOf(name) != target->kindOf(name)) {
          return false;
        }
        else {
          switch (source->kindOf(name)) {
            case AttributeKind::f:
              if (source->f(name) != target->f(name)) {
                return false;
              }
              break;
            case AttributeKind::fs:
              if (source->fs(name) != target->fs(name)) {
                return false;
              }
              break;
            case AttributeKind::i:
              if (source->i(name) != target->i(name)) {
                return false;
              }
              break;
            case AttributeKind::is:
              if (source->is(name) != target->is(name)) {
                return false;
              }
              break;
            case AttributeKind::s:
              if (source->s(name) != target->s(name)) {
                return false;
              }
              break;
            case AttributeKind::ss:
              if (source->ss(name) != target->ss(name)) {
                return false;
              }
              break;
            case AttributeKind::t:
            case AttributeKind::ts:
            case AttributeKind::g:
            case AttributeKind::gs:
              return false;
          }
        }
      }
    }
  }

  bool are_equal_inputs(ArrayRef<Value*> source, ArrayRef<Value*> target) {
    if (source.size() != target.size()) {
      return false;
    }

    return std::equal(source.begin(), source.end(), target.begin());
  }

  Node* find_candidate_node(Node* current, use_list uses) {
    for (auto use : uses) {
      if (use.user->kind() == current->kind() && use.user != current) {
        if (are_equal_inputs(current->inputs(), use.user->inputs())) {
          if (have_equal_attributes(current, use.user)) {
            return use.user;
          }
        }
      }
    }
    return nullptr;
  }

  bool runTransform(Node* node, Graph& graph, NodeDestroyType& destroy_current)
      override {
    auto candidate = find_candidate_node(node, node->inputs()[0]->uses());

    if (candidate) {
      auto n_outputs = node->outputs();
      auto c_outputs = candidate->outputs();

      for (size_t i = 0; i < n_outputs.size(); i += 1) {
        auto n_output = n_outputs[i];
        auto c_output = c_outputs[i];

        if (n_output->has_sizes()) {
          c_output->setSizes(n_output->sizes());
        }

        if (std::find(graph.outputs().rbegin(), graph.outputs().rend(),
                  n_output) != graph.outputs().rend()) {
          c_output->setUniqueName(n_output->uniqueName());
        }

        n_output->replaceAllUsesWith(c_output);
      }
      destroy_current = NodeDestroyType::DestroyOne;
      return true;
    }
    else {
      return false;
    }
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE