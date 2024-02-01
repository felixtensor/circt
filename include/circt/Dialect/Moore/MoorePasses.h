//===- Passes.h - Moore pass entry points -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MOOREPASSES_H
#define CIRCT_DIALECT_MOORE_MOOREPASSES_H

#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"
#include <memory>

namespace circt {
namespace moore {

class Declaration {
  // A map used to collect nets or variables and their values.
  // Only record the value produced by the last assignment op,
  // including the declaration assignment.
  DenseMap<Operation *, Value> netOrVarInfo;

  // Identifying an assignment statement whether been used by a net/variable.
  // If identified as true, delete it at the stage of conversion between
  // dialects, except for the output ports assignment.
  DenseMap<Operation *, bool> isUsedAssignment;

public:
  void addValue(Operation *op, Value value) { netOrVarInfo[op] = value; }
  void addIdentifier(Operation *op, bool b) {
    isUsedAssignment.insert({op, b});
  }

  auto getValue(Operation *op) { return netOrVarInfo.lookup(op); }
  auto getIdentifier(Operation *op) { return isUsedAssignment.lookup(op); }
};

extern Declaration decl;
std::unique_ptr<mlir::Pass> createMooreDeclarationsPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Moore/MoorePasses.h.inc"

} // namespace moore
} // namespace circt

#endif // CIRCT_DIALECT_MOORE_MOOREPASSES_H
