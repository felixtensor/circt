//===- MooreOps.td - Moore dialect operations --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MOOREOPS
#define CIRCT_DIALECT_MOORE_MOOREOPS

include "circt/Dialect/Moore/MooreTypes.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// Structure
//===----------------------------------------------------------------------===//

def SVModuleOp : MooreOp<"module", [
  IsolatedFromAbove,
  Symbol,
  RegionKindInterface,
  NoTerminator
]> {
  let summary = "A module definition";

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $sym_name attr-dict-with-keyword $body
  }];

  let extraClassDeclaration = [{
    static mlir::RegionKind getRegionKind(unsigned index) {
      return mlir::RegionKind::SSACFG;
    }

    mlir::Block &getBodyBlock() { return getBody().front(); }
  }];
}

def InstanceOp : MooreOp<"instance", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Create an instance of a module";

  let arguments = (ins StrAttr:$instanceName,
                       FlatSymbolRefAttr:$moduleName);
  let assemblyFormat = [{
    $instanceName $moduleName attr-dict
  }];
}

def Initial: I32EnumAttrCase<"Initial", 0, "initial">;
def Final: I32EnumAttrCase<"Final", 1, "final">;
def Always: I32EnumAttrCase<"Always", 2, "always">;
def AlwaysComb: I32EnumAttrCase<"AlwaysComb", 3, "always_comb">;
def AlwaysLatch: I32EnumAttrCase<"AlwaysLatch", 4, "always_latch">;
def AlwaysFF: I32EnumAttrCase<"AlwaysFF", 5, "always_ff">;

def ProcedureKindAttr: I32EnumAttr<"ProcedureKind", "Procedure kind",
            [Initial, Final, Always, AlwaysComb, AlwaysLatch, AlwaysFF]>{
  let cppNamespace = "circt::moore";
}

def ProcedureOp : MooreOp<"procedure", [
  SingleBlock,
  NoTerminator,
  NoRegionArguments,
  RecursiveMemoryEffects,
  RecursivelySpeculatable
]> {
  let summary = "A procedure executed at specific simulation steps";
  let description = [{
    See IEEE 1800-2017 § 9.2 "Structured procedures".
  }];

  let regions = (region SizedRegion<1>:$body);
  let arguments = (ins ProcedureKindAttr:$kind);
  let results = (outs);

  let assemblyFormat = [{
    $kind attr-dict-with-keyword $body
  }];

  let extraClassDeclaration = [{
    mlir::Block &getBodyBlock() {
      if (getBody().empty())
        getBody().emplaceBlock();
      return getBody().front();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Declarations
//===----------------------------------------------------------------------===//

def In: I32EnumAttrCase<"In", 0>;
def Out: I32EnumAttrCase<"Out", 1>;
def InOut: I32EnumAttrCase<"InOut", 2>;
def Ref: I32EnumAttrCase<"Ref", 3>;

def DirectionAttr: I32EnumAttr<"Direction", "Port direction",
                                  [In, Out, InOut, Ref]>{
  let cppNamespace = "circt::moore";
}

def PortOp : MooreOp<"port", [
  RecursiveMemoryEffects
]> {
  let summary = "Declare a port";
  let arguments = (ins StrAttr:$name, DirectionAttr:$direction);
  let results = (outs);
  let assemblyFormat = [{
    $direction $name attr-dict
  }];
}

def VariableOp : MooreOp<"variable", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  OptionalTypesMatchWith<"initial value and variable types match",
    "result", "initial", "$_self">,
]> {
  let summary = "A variable declaration";
  let description = [{
    See IEEE 1800-2017 § 6.8 "Variable declarations".
  }];
  let arguments = (ins StrAttr:$name, Optional<UnpackedType>:$initial);
  let results = (outs UnpackedType:$result);
  let assemblyFormat = [{
    `` custom<ImplicitSSAName>($name) ($initial^)? attr-dict
    `:` type($result)
  }];
}

def NetOp : MooreOp<"net", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  OptionalTypesMatchWith<"assigned value and variable types match",
    "result", "assignment", "$_self">,
]> {
  let summary = "A net declaration";
  let description = [{
    See IEEE 1800-2017 § 6.7 "Net declarations".
  }];
  let arguments = (ins
    StrAttr:$name,
    StrAttr:$kind,
    Optional<UnpackedType>:$assignment
  );
  let results = (outs UnpackedType:$result);
  let assemblyFormat = [{
    `` custom<ImplicitSSAName>($name) $kind ($assignment^)? attr-dict
    `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

def None: I32EnumAttrCase<"None", 0, "none">;
def PosEdge: I32EnumAttrCase<"PosEdge", 1, "posedge">;
def NegEdge: I32EnumAttrCase<"NegEdge", 2, "negedge">;
def BothEdges: I32EnumAttrCase<"BothEdges", 3, "bothedges">;

def EdgeAtrr: I32EnumAttr<"Edge", "Edge kind", 
                          [None, PosEdge, NegEdge, BothEdges]>{
  let cppNamespace = "circt::moore";
}

def EventControlOp : MooreOp<"event", [
  HasParent<"ProcedureOp">
]> {
  let summary = "Detecting posedge and negedge";
  let description = "See SV Spec 9.4.2.";
  let arguments = (ins EdgeAtrr:$edge, StrAttr:$name);
  let results = (outs);
  let assemblyFormat = [{
    $edge $name attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

def ConstantOp : MooreOp<"constant", [Pure]> {
  let summary = "A constant integer value";
  let arguments = (ins APIntAttr:$value);
  let results = (outs SimpleBitVectorType:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "Type":$type, "const APInt &":$value)>,
    OpBuilder<(ins "Type":$type, "int64_t":$value)>,
  ];
}

def ConversionOp : MooreOp<"conversion", [Pure]> {
  let summary = "A type conversion";
  let description = [{
    An explicit or implicit type conversion. These are either generated
    automatically in order to make assignments compatible:

    ```
    int a;
    shortint b;
    a = b;  // generates an implicit cast from shortint to int
    ```

    Or explicitly by the user through a type, sign, or const cast expression:

    ```
    byte'(a)
    unsigned'(a)
    signed'(a)
    42'(a)
    ```

    See IEEE 1800-2017 § 6.24 "Casting".
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs UnpackedType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

def NegOp : MooreOp<"neg", [Pure, SameOperandsAndResultType]> {
  let summary = "Arithmetic negation";
  let description = [{
    Negate a value to its two's complement form. If any bit in the input is Z or
    X, all bits in the result are set to X.

    See IEEE 1800-2017 § 11.4.3 "Arithmetic operators".
  }];
  let arguments = (ins SimpleBitVectorType:$input);
  let results = (outs SimpleBitVectorType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

def NotOp : MooreOp<"not", [Pure, SameOperandsAndResultType]> {
  let summary = "Bitwise unary negation";
  let description = [{
    Applies the boolean NOT operation to each bit in the input. Corresponds to
    the `~` operator, as well as the negation in the `~&`, `~|`, `^~`, and `~^`
    reduction operators.

    See IEEE 1800-2017 § 11.4.8 "Bitwise operators".

    | Input | Result |
    |-------|--------|
    | 0     | 1      |
    | 1     | 0      |
    | X     | X      |
    | Z     | X      |
  }];
  let arguments = (ins SimpleBitVectorType:$input);
  let results = (outs SimpleBitVectorType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

class ReduceOpBase<string mnemonic, string operatorName> : MooreOp<mnemonic, [
  Pure,
  TypesMatchWith<"result is single bit of input", "input", "result", [{
    $_self.cast<UnpackedType>()
      .getSimpleBitVector()
      .toSingleBit()
      .getType($_self.getContext())
  }]>
]> {
  let arguments = (ins SimpleBitVectorType:$input);
  let results = (outs SingleBitType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  let summary = !subst("$op", operatorName, "Reduction $op operator");
  let description = !subst("$op", operatorName, [{
    Reduces all bits in the input to a single result bit by iteratively applying
    the boolean $op operator. If the input has only a single bit, that bit is
    returned.

    See IEEE 1800-2017 § 11.4.9 "Reduction operators". See the corresponding
    `and`, `or`, and `xor` operations for the truth table.
  }]);
}

def ReduceAndOp : ReduceOpBase<"reduce_and", "AND">;
def ReduceOrOp : ReduceOpBase<"reduce_or", "OR">;
def ReduceXorOp : ReduceOpBase<"reduce_xor", "XOR">;

def BoolCastOp : MooreOp<"bool_cast", [
  Pure,
  TypesMatchWith<"result is single bit matching input domain",
    "input", "result", [{
    IntType::get($_self.getContext(),
      IntType::getAtomForDomain($_self.cast<UnpackedType>().getDomain()))
  }]>
]> {
  let summary = "Cast a value to a single bit boolean";
  let description = [{
    Convert a nonzero or true value into 1, a zero or false value into 0, and
    any value containing Z or X bits into a X. This conversion is useful in
    combination with the logical and, or, implication, equivalence, and negation
    operators.

    See IEEE 1800-2017 § 11.4.7 "Logical operators".
  }];
  let arguments = (ins UnpackedType:$input);
  let results = (outs SingleBitType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}


class BinaryOpBase<string mnemonic, list<Trait> traits = []> :
    MooreOp<mnemonic, traits # [Pure, SameOperandsAndResultType]> {
  let arguments = (ins SimpleBitVectorType:$lhs, SimpleBitVectorType:$rhs);
  let results = (outs SimpleBitVectorType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($result)
  }];
}

def AddOp : BinaryOpBase<"add", [Commutative]> {
  let summary = "Addition";
  let description = [{
    Add the operands. If any bit in the two operands is Z or X, all bits in the
    result are set to X.

    See IEEE 1800-2017 § 11.4.3 "Arithmetic operators".
  }];
}

def SubOp : BinaryOpBase<"sub"> {
  let summary = "Subtraction";
  let description = [{
    Subtract the right-hand side from the left-hand side operand. If any bit in
    the two operands is Z or X, all bits in the result are set to X.

    See IEEE 1800-2017 § 11.4.3 "Arithmetic operators".
  }];
}

def MulOp : BinaryOpBase<"mul", [Commutative]> {
  let summary = "Multiplication";
  let description = [{
    Multiply the operands. If any bit in the two operands is Z or X, all bits in
    the result are set to X.

    See IEEE 1800-2017 § 11.4.3 "Arithmetic operators".
  }];
}


#endif // CIRCT_DIALECT_MOORE_MOOREOPS
