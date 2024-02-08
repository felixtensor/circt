//===- MooreDeclarations.cpp - Collect net/variable declarations ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MooreDeclarations pass.
// Use to collect net/variable declarations and bound a value to them.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/Moore/MoorePasses.h"

using namespace circt;
using namespace moore;

namespace {
struct MooreDeclarationsPass
    : public MooreDeclarationsBase<MooreDeclarationsPass> {
  void runOnOperation() override;
};
} // namespace

extern Declaration moore::decl;
void MooreDeclarationsPass::runOnOperation() {

  getOperation()->walk([&](SVModuleOp moduleOp) {
    for (auto &op : moduleOp.getOps()) {

      TypeSwitch<Operation *, void>(&op)

          .Case<VariableOp, NetOp>([&](auto &op) {
            auto operandIt = op.getOperands();
            auto value = operandIt.empty() ? nullptr : op.getOperand(0);
            decl.addValue(op, value);
          })
          .Case<CAssignOp, BPAssignOp, PAssignOp, PCAssignOp>([&](auto &op) {
            auto destOp = op.getOperand(0).getDefiningOp();
            auto srcValue = op.getOperand(1);
            decl.addValue(destOp, srcValue);
            decl.addIdentifier(op, true);
          })
          .Case<PortOp>([&](auto &op) { decl.addValue(op, nullptr); });
    };
    return WalkResult::advance();
  });
  //   markAllAnalysesPreserved();
}

std::unique_ptr<mlir::Pass> circt::moore::createMooreDeclarationsPass() {
  return std::make_unique<MooreDeclarationsPass>();
}
