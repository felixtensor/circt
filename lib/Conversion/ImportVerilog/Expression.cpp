//===- Statement.cpp - Slang expression conversion ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/ast/ASTVisitor.h"
#include "slang/ast/Symbol.h"
#include "slang/ast/symbols/CompilationUnitSymbols.h"
#include "slang/ast/symbols/InstanceSymbols.h"
#include "slang/ast/symbols/VariableSymbols.h"
#include "slang/ast/types/AllTypes.h"
#include "slang/ast/types/Type.h"
#include "slang/syntax/SyntaxVisitor.h"

using namespace circt;
using namespace ImportVerilog;

namespace {
struct ExprVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;

  Value visit(const slang::ast::NamedValueExpression &expr) {
    // TODO: This needs something more robust. Slang should have resolved names
    // already. Better use those pointers instead of names.
    return context.varSymbolTable.lookup(expr.getSymbolReference()->name);
  }

  Value visit(const slang::ast::ConversionExpression &expr) {
    auto type = context.convertType(*expr.type);
    auto operand = context.convertExpression(expr.operand());
    if (!operand)
      return {};
    return builder.create<moore::ConversionOp>(loc, type, operand);
  }

  Value visit(const slang::ast::AssignmentExpression &expr) {
    auto lhs = context.convertExpression(expr.left());
    auto rhs = context.convertExpression(expr.right());
    if (!lhs || !rhs)
      return {};
    if (lhs.getType() != rhs.getType())
      rhs = builder.create<moore::ConversionOp>(loc, lhs.getType(), rhs);
    if (expr.isNonBlocking())
      builder.create<moore::PAssignOp>(loc, lhs, rhs);
    else if (expr.syntax->parent->kind ==
             slang::syntax::SyntaxKind::ContinuousAssign)
      builder.create<moore::CAssignOp>(loc, lhs, rhs);
    else if (expr.syntax->parent->kind ==
             slang::syntax::SyntaxKind::ProceduralAssignStatement)
      builder.create<moore::PCAssignOp>(loc, lhs, rhs);
    else
      builder.create<moore::BPAssignOp>(loc, lhs, rhs);
    return lhs;
  }

  Value visit(const slang::ast::UnaryExpression &expr) {
    auto arg = context.convertExpression(expr.operand());

    switch (expr.op) {
    case slang::ast::UnaryOperator::Plus:
      return builder.create<moore::UnaryOp>(loc, moore::Unary::Plus, arg);
    case slang::ast::UnaryOperator::Minus:
      return builder.create<moore::UnaryOp>(loc, moore::Unary::Minus, arg);
    case slang::ast::UnaryOperator::BitwiseNot:
      return builder.create<moore::ReductionOp>(
          loc, moore::Reduction::BitwiseNot, arg);
    case slang::ast::UnaryOperator::BitwiseAnd:
      return builder.create<moore::ReductionOp>(
          loc, moore::Reduction::BitwiseAnd, arg);
    case slang::ast::UnaryOperator::BitwiseOr:
      return builder.create<moore::ReductionOp>(
          loc, moore::Reduction::BitwiseOr, arg);
    case slang::ast::UnaryOperator::BitwiseXor:
      return builder.create<moore::ReductionOp>(
          loc, moore::Reduction::BitwiseXor, arg);
    case slang::ast::UnaryOperator::BitwiseNand:
      return builder.create<moore::ReductionOp>(
          loc, moore::Reduction::BitwiseNand, arg);
    case slang::ast::UnaryOperator::BitwiseNor:
      return builder.create<moore::ReductionOp>(
          loc, moore::Reduction::BitwiseNor, arg);
    case slang::ast::UnaryOperator::BitwiseXnor:
      return builder.create<moore::ReductionOp>(
          loc, moore::Reduction::BitwiseXnor, arg);
    case slang::ast::UnaryOperator::LogicalNot:
      return builder.create<moore::UnaryOp>(loc, moore::Unary::LogicalNot, arg);
    case slang::ast::UnaryOperator::Preincrement:
    case slang::ast::UnaryOperator::Predecrement:
    case slang::ast::UnaryOperator::Postincrement:
    case slang::ast::UnaryOperator::Postdecrement:

    default:
      mlir::emitError(loc, "unsupported unary operator");
      return {};
    }
  }

  Value visit(const slang::ast::BinaryExpression &expr) {
    auto lhs = context.convertExpression(expr.left());
    auto rhs = context.convertExpression(expr.right());
    if (!lhs || !rhs)
      return {};

    switch (expr.op) {
    case slang::ast::BinaryOperator::Add:
      return builder.create<moore::AddOp>(loc, lhs, rhs);
    case slang::ast::BinaryOperator::Subtract:
      mlir::emitError(loc, "unsupported binary operator: subtract");
      return {};
    case slang::ast::BinaryOperator::Multiply:
      return builder.create<moore::MulOp>(loc, lhs, rhs);
    case slang::ast::BinaryOperator::Divide:
      mlir::emitError(loc, "unsupported binary operator: divide");
      return {};
    case slang::ast::BinaryOperator::Mod:
      mlir::emitError(loc, "unsupported binary operator: mod");
      return {};
    case slang::ast::BinaryOperator::BinaryAnd:
      return builder.create<moore::BitwiseOp>(loc, moore::Bitwise::BinaryAnd,
                                              lhs, rhs);
    case slang::ast::BinaryOperator::BinaryOr:
      return builder.create<moore::BitwiseOp>(loc, moore::Bitwise::BinaryOr,
                                              lhs, rhs);
    case slang::ast::BinaryOperator::BinaryXor:
      return builder.create<moore::BitwiseOp>(loc, moore::Bitwise::BinaryXor,
                                              lhs, rhs);
    case slang::ast::BinaryOperator::BinaryXnor:
      return builder.create<moore::BitwiseOp>(loc, moore::Bitwise::BinaryXnor,
                                              lhs, rhs);
    case slang::ast::BinaryOperator::Equality:
      return builder.create<moore::EqualityOp>(loc, lhs, rhs);
    case slang::ast::BinaryOperator::Inequality:
      return builder.create<moore::InEqualityOp>(loc, lhs, rhs);
    case slang::ast::BinaryOperator::CaseEquality:
      return builder.create<moore::EqualityOp>(loc, lhs, rhs,
                                               builder.getUnitAttr());
    case slang::ast::BinaryOperator::CaseInequality:
      return builder.create<moore::InEqualityOp>(loc, lhs, rhs,
                                                 builder.getUnitAttr());
    case slang::ast::BinaryOperator::GreaterThanEqual:
      // TODO: I think should integrate these four relation operators into one
      // builder.create. But I failed, the error is `resultNumber <
      // getNumResults() && ... ` from Operation.h:983.
      return builder.create<moore::RelationalOp>(
          loc, moore::Relation::GreaterThanEqual, lhs, rhs);
    case slang::ast::BinaryOperator::GreaterThan:
      return builder.create<moore::RelationalOp>(
          loc, moore::Relation::GreaterThan, lhs, rhs);
    case slang::ast::BinaryOperator::LessThanEqual:
      return builder.create<moore::RelationalOp>(
          loc, moore::Relation::LessThanEqual, lhs, rhs);
    case slang::ast::BinaryOperator::LessThan:
      return builder.create<moore::RelationalOp>(loc, moore::Relation::LessThan,
                                                 lhs, rhs);
    case slang::ast::BinaryOperator::WildcardEquality:
      mlir::emitError(loc, "unsupported binary operator: wildcard equality");
      return {};
    case slang::ast::BinaryOperator::WildcardInequality:
      mlir::emitError(loc, "unsupported binary operator: wildcard inequality");
      return {};
    case slang::ast::BinaryOperator::LogicalAnd:
      return builder.create<moore::LogicalOp>(loc, moore::Logic::LogicalAnd,
                                              lhs, rhs);
    case slang::ast::BinaryOperator::LogicalOr:
      return builder.create<moore::LogicalOp>(loc, moore::Logic::LogicalOr, lhs,
                                              rhs);
    case slang::ast::BinaryOperator::LogicalImplication:
      return builder.create<moore::LogicalOp>(
          loc, moore::Logic::LogicalImplication, lhs, rhs);
    case slang::ast::BinaryOperator::LogicalEquivalence:
      return builder.create<moore::LogicalOp>(
          loc, moore::Logic::LogicalEquivalence, lhs, rhs);
    case slang::ast::BinaryOperator::LogicalShiftLeft:
      return builder.create<moore::ShlOp>(loc, lhs, rhs);
    case slang::ast::BinaryOperator::LogicalShiftRight:
      return builder.create<moore::ShrOp>(loc, lhs, rhs);
    case slang::ast::BinaryOperator::ArithmeticShiftLeft:
      return builder.create<moore::ShlOp>(loc, lhs, rhs, builder.getUnitAttr());
    case slang::ast::BinaryOperator::ArithmeticShiftRight:
      return builder.create<moore::ShrOp>(loc, lhs, rhs, builder.getUnitAttr());
    case slang::ast::BinaryOperator::Power:
      mlir::emitError(loc, "unsupported binary operator: power");
      return {};

    default:
      mlir::emitError(loc, "unsupported binary operator");
      return {};
    }
  }

  Value visit(const slang::ast::IntegerLiteral &expr) {
    // TODO: This is wildly unsafe and breaks for anything larger than 32 bits.
    auto value = expr.getValue().as<uint32_t>().value();
    auto type = context.convertType(*expr.type);
    return builder.create<moore::ConstantOp>(loc, type, value);
  }

  Value visit(const slang::ast::ConcatenationExpression &expr) {
    SmallVector<Value> operands;
    for (auto *operand : expr.operands()) {
      auto value = context.convertExpression(*operand);
      if (!value)
        return {};
      operands.push_back(value);
    }
    return builder.create<moore::ConcatOp>(loc, operands);
  }

  /// Emit an error for all other expressions.
  template <typename T>
  Value visit(T &&node) {
    mlir::emitError(loc, "unsupported expression: ")
        << slang::ast::toString(node.kind);
    return {};
  }

  Value visitInvalid(const slang::ast::Expression &expr) {
    mlir::emitError(loc, "invalid expression");
    return {};
  }
};
} // namespace

Value Context::convertExpression(const slang::ast::Expression &expr) {
  auto loc = convertLocation(expr.sourceRange.start());
  return expr.visit(ExprVisitor{*this, loc, builder});
}
