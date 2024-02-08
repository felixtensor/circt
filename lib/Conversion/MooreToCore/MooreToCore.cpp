//===- MooreToCore.cpp - Moore To Core Conversion Pass --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main Moore to Core Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/MooreToCore.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MoorePasses.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace circt;
using namespace moore;

Declaration moore::decl;

MoorePortInfo::MoorePortInfo(moore::SVModuleOp moduleOp) {
  SmallVector<hw::PortInfo, 4> inputs, outputs;

  // Gather all input or output ports.
  for (auto portOp : moduleOp.getBodyBlock().getOps<PortOp>()) {
    auto portName = portOp.getNameAttr();
    auto portLoc = portOp.getLoc();
    auto portTy = portOp.getType();
    auto argNum = inputs.size();

    switch (portOp.getDirection()) {
    case Direction::In:
      inputs.push_back(
          hw::PortInfo{{portName, portTy, hw::ModulePort::Direction::Input},
                       argNum,
                       {},
                       portLoc});
      inputsPort[portName] = std::make_pair(portOp, portTy);
      break;
    case Direction::InOut:
      inputs.push_back(hw::PortInfo{{portName, hw::InOutType::get(portTy),
                                     hw::ModulePort::Direction::InOut},
                                    argNum,
                                    {},
                                    portLoc});
      inputsPort[portName] = std::make_pair(portOp, portTy);
      break;
    case Direction::Out:
      if (!portOp->getUsers().empty())
        outputs.push_back(
            hw::PortInfo{{portName, portTy, hw::ModulePort::Direction::Output},
                         argNum,
                         {},
                         portOp.getLoc()});
      outputsPort[portName] = std::make_pair(portOp, portTy);
      break;
    case Direction::Ref:
      // TODO: Support parsing Direction::Ref port into portInfo structure.
      break;
    }
  }
  hwPorts = std::make_unique<hw::ModulePortInfo>(inputs, outputs);
}

namespace {

/// Zero-extends if it is too narrow.
/// Truncates if the integer is too wide and the truncated part is zero, if it
/// is not zero it returns the max value integer of target-width.
static Value adjustIntegerWidth(OpBuilder &builder, Value value,
                                uint32_t targetWidth, Location loc) {
  uint32_t intWidth = value.getType().getIntOrFloatBitWidth();
  if (intWidth == targetWidth)
    return value;

  if (intWidth < targetWidth) {
    Value zeroExt = builder.create<hw::ConstantOp>(
        loc, builder.getIntegerType(targetWidth - intWidth), 0);
    return builder.create<comb::ConcatOp>(loc, ValueRange{zeroExt, value});
  }

  Value hi = builder.create<comb::ExtractOp>(loc, value, targetWidth,
                                             intWidth - targetWidth);
  Value zero = builder.create<hw::ConstantOp>(
      loc, builder.getIntegerType(intWidth - targetWidth), 0);
  Value isZero = builder.create<comb::ICmpOp>(loc, comb::ICmpPredicate::eq, hi,
                                              zero, false);
  Value lo = builder.create<comb::ExtractOp>(loc, value, 0, targetWidth);
  Value max = builder.create<hw::ConstantOp>(
      loc, builder.getIntegerType(targetWidth), -1);
  return builder.create<comb::MuxOp>(loc, isZero, lo, max, false);
}

/// Due to the result type of the `lt`, or `le`, or `gt`, or `ge` ops are
/// always unsigned, estimating their operands type.
static bool isSignedType(Operation *op) {
  return TypeSwitch<Operation *, bool>(op)
      .template Case<LtOp, LeOp, GtOp, GeOp>([&](auto op) -> bool {
        return cast<UnpackedType>(op->getOperand(0).getType())
                   .castToSimpleBitVector()
                   .isSigned() &&
               cast<UnpackedType>(op->getOperand(1).getType())
                   .castToSimpleBitVector()
                   .isSigned();
      })
      .Default([&](auto op) -> bool {
        return cast<UnpackedType>(op->getResult(0).getType())
            .castToSimpleBitVector()
            .isSigned();
      });
}

/// Not define the predicate for `relation` and `equality` operations in the
/// MooreDialect, but comb needs it. Return a correct `comb::ICmpPredicate`
/// corresponding to different moore `relation` and `equality` operations.
static comb::ICmpPredicate getCombPredicate(Operation *op) {
  using comb::ICmpPredicate;
  return TypeSwitch<Operation *, ICmpPredicate>(op)
      .Case<LtOp>([&](auto op) {
        return isSignedType(op) ? ICmpPredicate::slt : ICmpPredicate::ult;
      })
      .Case<LeOp>([&](auto op) {
        return isSignedType(op) ? ICmpPredicate::sle : ICmpPredicate::ule;
      })
      .Case<GtOp>([&](auto op) {
        return isSignedType(op) ? ICmpPredicate::sgt : ICmpPredicate::ugt;
      })
      .Case<GeOp>([&](auto op) {
        return isSignedType(op) ? ICmpPredicate::sge : ICmpPredicate::uge;
      })
      .Case<EqOp>([&](auto op) { return ICmpPredicate::eq; })
      .Case<NeOp>([&](auto op) { return ICmpPredicate::ne; })
      .Case<CaseEqOp>([&](auto op) { return ICmpPredicate::ceq; })
      .Case<CaseNeOp>([&](auto op) { return ICmpPredicate::cne; })
      .Case<WildcardEqOp>([&](auto op) { return ICmpPredicate::weq; })
      .Case<WildcardNeOp>([&](auto op) { return ICmpPredicate::wne; });
}

//===----------------------------------------------------------------------===//
// Structure Conversion
//===----------------------------------------------------------------------===//

struct SVModuleOpConv : public OpConversionPattern<SVModuleOp> {
  SVModuleOpConv(TypeConverter &typeConverter, MLIRContext *ctx,
                 MoorePortInfoMap &portMap)
      : OpConversionPattern<SVModuleOp>(typeConverter, ctx), portMap(portMap) {}
  LogicalResult
  matchAndRewrite(SVModuleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    const circt::MoorePortInfo &mp = portMap.at(op.getSymNameAttr());

    // Create the hw.module to replace svmoduleOp
    auto hwModuleOp = rewriter.create<hw::HWModuleOp>(
        op.getLoc(), op.getSymNameAttr(), *mp.hwPorts);
    rewriter.eraseBlock(hwModuleOp.getBodyBlock());
    rewriter.inlineRegionBefore(*op.getBodyBlock().getParent(),
                                hwModuleOp.getBodyRegion(),
                                hwModuleOp.getBodyRegion().end());
    auto *hwBody = hwModuleOp.getBodyBlock();

    // Replace all relating logic of input port definitions for the input
    // block arguments. And update relating uses chain.
    for (auto [index, input] : llvm::enumerate(mp.hwPorts->getInputs())) {
      BlockArgument newArg;
      auto portOp = mp.inputsPort.at(input.name).first;
      auto inputTy = mp.inputsPort.at(input.name).second;
      rewriter.modifyOpInPlace(hwModuleOp, [&]() {
        newArg = hwBody->addArgument(inputTy, portOp.getLoc());
      });
      rewriter.replaceAllUsesWith(portOp->getResults(), newArg);
    }

    // Adjust all relating logic of output port definitions for rewriting
    // hw.output op.
    SmallVector<Value> outputValues;
    for (auto [index, output] : llvm::enumerate(mp.hwPorts->getOutputs())) {
      auto portOp = mp.outputsPort.at(output.name).first;
      outputValues.push_back(portOp->getResult(0));
    }

    // Rewrite the hw.output op
    rewriter.setInsertionPointToEnd(hwBody);
    rewriter.create<hw::OutputOp>(op.getLoc(), outputValues);

    // Erase the original op
    rewriter.eraseOp(op);
    return success();
  }
  MoorePortInfoMap &portMap;
};

struct ProcedureOpConv : public OpConversionPattern<ProcedureOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ProcedureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    switch (adaptor.getKind()) {
    case ProcedureKind::AlwaysComb:
      rewriter.setInsertionPointAfter(op->getPrevNode());
      rewriter.inlineBlockBefore(&op.getBodyBlock(),
                                 op->getPrevNode()->getBlock(),
                                 rewriter.getInsertionPoint());
      rewriter.eraseOp(op);
      return success();
    case ProcedureKind::Always:
    case ProcedureKind::AlwaysFF:
    case ProcedureKind::AlwaysLatch:
    case ProcedureKind::Initial:
    case ProcedureKind::Final:
      return emitError(op->getLoc(), "Unsupported procedure operation");
    };
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Declaration Conversion
//===----------------------------------------------------------------------===//

template <typename OpTy>
struct DeclOpConv : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;
  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder builder(op->getLoc(), op.getContext());
    Value value = decl.getValue(op);
    if (!value) {
      rewriter.eraseOp(op);
      return success();
    }

    Type resultType =
        ConversionPattern::typeConverter->convertType(op.getResult().getType());
    ConversionPattern::typeConverter->materializeTargetConversion(
        builder, op->getLoc(), resultType, {value});
    rewriter.replaceOpWithNewOp<hw::WireOp>(op, value, op.getNameAttr());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Expression Conversion
//===----------------------------------------------------------------------===//

struct ConstantOpConv : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, op.getValueAttr());
    return success();
  }
};

struct ConcatOpConv : public OpConversionPattern<ConcatOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<comb::ConcatOp>(op, adaptor.getValues());
    return success();
  }
};

struct ShlOpConv : public OpConversionPattern<ShlOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ShlOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());

    // Comb shift operations require the same bit-width for value and amount
    Value amount =
        adjustIntegerWidth(rewriter, adaptor.getAmount(),
                           resultType.getIntOrFloatBitWidth(), op->getLoc());
    rewriter.replaceOpWithNewOp<comb::ShlOp>(op, resultType, adaptor.getValue(),
                                             amount, false);
    return success();
  }
};

struct ShrOpConv : public OpConversionPattern<ShrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ShrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());

    // Comb shift operations require the same bit-width for value and amount
    Value amount =
        adjustIntegerWidth(rewriter, adaptor.getAmount(),
                           resultType.getIntOrFloatBitWidth(), op->getLoc());

    if (adaptor.getArithmetic() && isSignedType(op)) {
      rewriter.replaceOpWithNewOp<comb::ShrSOp>(
          op, resultType, adaptor.getValue(), amount, false);
      return success();
    }

    rewriter.replaceOpWithNewOp<comb::ShrUOp>(
        op, resultType, adaptor.getValue(), amount, false);
    return success();
  }
};

template <typename OpTy>
struct UnaryOpConv : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<comb::ParityOp>(op, adaptor.getInput());
    return success();
  }
};

struct NotOpConv : public OpConversionPattern<NotOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        ConversionPattern::typeConverter->convertType(op.getResult().getType());
    Value max = rewriter.create<hw::ConstantOp>(op.getLoc(), resultType, -1);
    rewriter.replaceOpWithNewOp<comb::XorOp>(op, adaptor.getInput(), max);
    return success();
  }
};

template <typename SourceOp, typename TargetOp>
struct BinaryOpConv : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<DivOp>(op) && isSignedType(op)) {
      rewriter.replaceOpWithNewOp<comb::DivSOp>(op, adaptor.getLhs(),
                                                adaptor.getRhs(), false);
      return success();
    }
    if (isa<ModOp>(op) && isSignedType(op)) {
      rewriter.replaceOpWithNewOp<comb::ModSOp>(op, adaptor.getLhs(),
                                                adaptor.getRhs(), false);
      return success();
    }

    rewriter.replaceOpWithNewOp<TargetOp>(op, adaptor.getLhs(),
                                          adaptor.getRhs(), false);
    return success();
  }
};

template <typename SourceOp>
struct ICmpOpConv : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adapter,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        ConversionPattern::typeConverter->convertType(op.getResult().getType());
    comb::ICmpPredicate pred = getCombPredicate(op);

    rewriter.replaceOpWithNewOp<comb::ICmpOp>(
        op, resultType, pred, adapter.getLhs(), adapter.getRhs());
    return success();
  }
};

struct ExtractOpConv : public OpConversionPattern<ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value value = adaptor.getInput();
    uint32_t lowBit = adaptor.getLowBit();
    uint32_t width = cast<UnpackedType>(op.getResult().getType())
                         .castToSimpleBitVectorOrNull()
                         .size;

    rewriter.replaceOpWithNewOp<comb::ExtractOp>(op, value, lowBit, width);
    return success();
  }
};

struct ConversionOpConv : public OpConversionPattern<ConversionOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ConversionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    Value amount =
        adjustIntegerWidth(rewriter, adaptor.getInput(),
                           resultType.getIntOrFloatBitWidth(), op->getLoc());
    rewriter.replaceOpWithNewOp<hw::BitcastOp>(op, resultType, amount);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Statement Conversion
//===----------------------------------------------------------------------===//

/// TODO: The `PAssign`, `PCAssign` ops need to convert.
template <typename SourceOp>
struct AssignOpConv : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.eraseOp(op);
    return success();
  }
};

struct HWOutputOpConv : public OpConversionPattern<hw::OutputOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::OutputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<hw::OutputOp>(op, adaptor.getOperands());
    return success();
  }
};

struct CondBranchOpConv : public OpConversionPattern<cf::CondBranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, adaptor.getCondition(), adaptor.getTrueDestOperands(),
        adaptor.getFalseDestOperands(), op.getTrueDest(), op.getFalseDest());
    return success();
  }
};

struct BranchOpConv : public OpConversionPattern<cf::BranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, op.getDest(),
                                              adaptor.getDestOperands());
    return success();
  }
};

struct UnrealizedConversionCastConv
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> convResTypes;
    if (typeConverter->convertTypes(op.getResultTypes(), convResTypes).failed())
      return failure();

    // Drop the cast if the operand and result types agree after type
    // conversion.
    if (convResTypes == adaptor.getOperands().getTypes()) {
      rewriter.replaceOp(op, adaptor.getOperands());
      return success();
    }

    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, convResTypes, adaptor.getOperands());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Conversion Infrastructure
//===----------------------------------------------------------------------===//

static bool isMooreType(Type type) { return type.isa<UnpackedType>(); }

static bool hasMooreType(TypeRange types) {
  return llvm::any_of(types, isMooreType);
}

static bool hasMooreType(ValueRange values) {
  return hasMooreType(values.getTypes());
}

static void updateMoorePortUseChain(PortOp op) {
  switch (op.getDirection()) {
  // The users of In / InOut direction port are replaced in the HWModuleOp's
  // generation. When it has no users left -> Erase the op, otherwise throw
  // failure messages.
  case Direction::In:
  case Direction::InOut:
    if (op->getUsers().empty())
      op.erase();
    break;

    // Handle the users of Out direction port, skip bpassign & passign op.
  case Direction::Out:
    // FIXME: The handling of different assignments here is a bit rough.
    op->getResult(0).replaceAllUsesWith(decl.getValue(op));
    op.erase();
    break;

    // TODO: Support converting port operation of Ref direction.
  case Direction::Ref:
    op.emitOpError("Not supported conversion of direction [")
        << op.getDirectionAttr() << "] port operation.";
  }
}

template <typename Op>
static void addGenericLegality(ConversionTarget &target) {
  target.addDynamicallyLegalOp<Op>([](Op op) {
    return !hasMooreType(op->getOperands()) && !hasMooreType(op->getResults());
  });
}

static void populateLegality(ConversionTarget &target,
                             bool isConvertStructure) {
  if (isConvertStructure) {
    target.addLegalDialect<MooreDialect>();
    target.addIllegalOp<moore::SVModuleOp>();
  } else {
    target.addIllegalDialect<MooreDialect>();
    target.addDynamicallyLegalOp<hw::HWModuleOp>([](hw::HWModuleOp op) {
      return !hasMooreType(op.getInputTypes()) &&
             !hasMooreType(op.getOutputTypes()) &&
             !hasMooreType(op.getBody().getArgumentTypes());
    });
    target.addDynamicallyLegalOp<hw::OutputOp>(
        [](hw::OutputOp op) { return !hasMooreType(op.getOutputs()); });
    target.addDynamicallyLegalOp<hw::InstanceOp>([](hw::InstanceOp op) {
      return !hasMooreType(op.getInputs()) &&
             !hasMooreType(op->getResultTypes());
    });
  }

  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<comb::CombDialect>();
  target.addLegalDialect<mlir::BuiltinDialect>();
  target.addLegalDialect<scf::SCFDialect>();
  addGenericLegality<cf::CondBranchOp>(target);
  addGenericLegality<cf::BranchOp>(target);
  addGenericLegality<UnrealizedConversionCastOp>(target);
}

static void populateTypeConversion(TypeConverter &typeConverter) {
  // Directly map simple bit vector types to a compact integer type. This needs
  // to be added after all of the other conversions above, such that SBVs
  // conversion gets tried first before any of the others.
  typeConverter.addConversion([&](UnpackedType type) -> std::optional<Type> {
    if (auto sbv = type.getSimpleBitVectorOrNull())
      return mlir::IntegerType::get(type.getContext(), sbv.size);
    return std::nullopt;
  });

  // When converting the net/variable declarations without initial value, it
  // will apply an unconverted value. Example `logic a; assign a = 1'b1;`.
  // And fail to convert the Moore type to the legal type, if directly call
  // `materializeTargetConversion()`.
  typeConverter.addTargetMaterialization(
      [&](OpBuilder &builder, Type type, ValueRange values, Location loc) {
        assert(values.size() == 1);
        values[0].setType(typeConverter.convertType(type));
        return values[0];
      });

  // Valid target types.
  typeConverter.addConversion([](mlir::IntegerType type) { return type; });

  // Materialize target types
  typeConverter.addTargetMaterialization(
      [&](OpBuilder &builder, Type type, ValueRange values, Location loc) {
        assert(values.size() == 1);
        values[0].setType(typeConverter.convertType(type));
        return values[0];
      });
}

void circt::populateMooreStructureConversionPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    circt::MoorePortInfoMap &portInfoMap) {
  auto *context = patterns.getContext();

  patterns.add<SVModuleOpConv>(typeConverter, context, portInfoMap);
}

void circt::populateMooreToCoreConversionPatterns(TypeConverter &typeConverter,
                                                  RewritePatternSet &patterns) {
  auto *context = patterns.getContext();

  patterns
      .add<ProcedureOpConv, DeclOpConv<NetOp>, DeclOpConv<VariableOp>,
           ConstantOpConv, ConcatOpConv, ShlOpConv, ShrOpConv,
           UnaryOpConv<ReduceAndOp>, UnaryOpConv<ReduceOrOp>,
           UnaryOpConv<ReduceXorOp>, UnaryOpConv<BoolCastOp>, NotOpConv,
           BinaryOpConv<AddOp, comb::AddOp>, BinaryOpConv<SubOp, comb::SubOp>,
           BinaryOpConv<MulOp, comb::MulOp>, BinaryOpConv<DivOp, comb::DivUOp>,
           BinaryOpConv<ModOp, comb::ModUOp>, BinaryOpConv<AndOp, comb::AndOp>,
           BinaryOpConv<OrOp, comb::OrOp>, BinaryOpConv<XorOp, comb::XorOp>,
           ICmpOpConv<LtOp>, ICmpOpConv<LeOp>, ICmpOpConv<GtOp>,
           ICmpOpConv<GeOp>, ICmpOpConv<EqOp>, ICmpOpConv<NeOp>,
           ICmpOpConv<CaseEqOp>, ICmpOpConv<CaseNeOp>, ICmpOpConv<WildcardEqOp>,
           ICmpOpConv<WildcardNeOp>, ExtractOpConv, ConversionOpConv,
           HWOutputOpConv, AssignOpConv<CAssignOp>, AssignOpConv<BPAssignOp>,
           CondBranchOpConv, BranchOpConv, UnrealizedConversionCastConv>(
          typeConverter, context);

  hw::populateHWModuleLikeTypeConversionPattern(
      hw::HWModuleOp::getOperationName(), patterns, typeConverter);
}

//===----------------------------------------------------------------------===//
// Moore to Core Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct MooreToCorePass : public ConvertMooreToCoreBase<MooreToCorePass> {
  void runOnOperation() override;
};
} // namespace

/// Create a Moore to core dialects conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> circt::createConvertMooreToCorePass() {
  return std::make_unique<MooreToCorePass>();
}

/// This is the main entrypoint for the Moore to Core conversion pass.
void MooreToCorePass::runOnOperation() {
  MLIRContext &context = getContext();
  ModuleOp module = getOperation();

  // This is used to collect the net/variable/port declarations with their
  // assigned value and identify the assignment statements that had been used.
  auto pm = PassManager::on<ModuleOp>(&context);
  pm.addPass(moore::createMooreDeclarationsPass());
  if (failed(pm.run(module)))
    return signalPassFailure();

  ConversionTarget target(context);
  TypeConverter typeConverter;
  RewritePatternSet patterns(&context);

  // Generate moore module signatures.
  MoorePortInfoMap portInfoMap;
  for (auto svModuleOp : module.getOps<SVModuleOp>())
    portInfoMap.try_emplace(svModuleOp.getSymNameAttr(),
                            MoorePortInfo(svModuleOp));

  // First to convert structral moore operations to hw.
  populateTypeConversion(typeConverter);
  populateLegality(target, true);
  populateMooreStructureConversionPatterns(typeConverter, patterns,
                                           portInfoMap);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
  patterns.clear();

  // Ensure that use chains of port operations have been updated and all port
  // operations have been handled.
  module->walk([&](PortOp op) { updateMoorePortUseChain(op); });

  // Second to convert miscellaneous moore operations to core ir.
  populateLegality(target, false);
  populateMooreToCoreConversionPatterns(typeConverter, patterns);

  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
