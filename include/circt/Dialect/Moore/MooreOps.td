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
include "mlir/IR/OpBase.td"
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
  let summary = "A systemVerilog module";
  let description = [{
    The `moore.module` operation represents a sv module. It is logically
    equivalent to the module in SystemVerilog.

    Example:
    ```mlir
      moore.module @top {
        %a = moore.port In : !moore.logic
        %b = moore.port In : !moore.logic
        %c = moore.port Out : !moore.logic
        %0 = moore.mir.shl %a, %b : !moore.logic, !moore.logic
        moore.mir.cassign %c, %0 : !moore.logic
      }
    ```
  }];

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
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
]> {
  let summary = "Create an instance of a module";
  let description = [{
    This represents an instance of a module. The inputs and results are
    the inputs and results of module. The inputs and results types of the
    instance must match the specified module type.
 
    Example:
    ```mlir
    moore.instance "foo" @Foo() -> () : () -> ()
    moore.instance "bar" @Bar(%a) -> (%b) : (!moore.logic) -> (!moore.logic)
    ```
  }];
  let arguments = (ins StrAttr:$instanceName,
                       FlatSymbolRefAttr:$moduleName,
                       Variadic<AnyType>:$inputs,
                       Variadic<AnyType>:$outputs);
  let results = (outs);

  let assemblyFormat = [{
    $instanceName $moduleName `(` $inputs `)` `->` `(` $outputs `)` attr-dict `:`
    functional-type($inputs, $outputs)
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
    The procedure opreation has five kinds: initial, final, always, always_comb,
    always_latch, always_ff. See IEEE 1800-2017 § 9.2 "Structured procedures".

    Example:
    ```mlir
      moore.procedure initial {}
      moore.procedure final {}
      moore.procedure always {}
      moore.procedure always_comb {}
      moore.procedure always_latch {}
      moore.procedure always_ff {}
    ```
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
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  RecursiveMemoryEffects
]> {
  let summary = "Declare a port";
  let description = [{
    The `moore.port` operation defines declaration of ports from original module
    / interface / instance ... It contains port direction (four types: 
    **In**, **Out**, **InOut**, **Ref**); port type and port name.

    Example:
    ```mlir
      %a = moore.port In : !moore.logic
      %b = moore.port Out : !moore.logic
    ```
  }];

  let arguments = (ins StrAttr:$name, DirectionAttr:$direction);
  let results = (outs UnpackedType:$result);
  let assemblyFormat = [{
    `` custom<ImplicitSSAName>($name) $direction attr-dict
    `:` type($result)
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

    Example:
    ```mlir
      %var1 = moore.variable : !moore.int
      %var2 = moore.variable %var1 : !moore.int
      %d0 = moore.variable : !moore.logic
      %e2 = moore.variable : !moore.packed<named<"myEnum", enum<shortint, loc(
    ```
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
    See IEEE 1800-2017 § 6.7 "Net declarations.

    Example:
    ```mlir
      %d0 = moore.net "wire" : !moore.logic
      %d1 = moore.net "tri" : !moore.logic
    ```
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
  let description = [{
    See SV Spec 9.4.2.

    Example:
    ```mlir
      moore.event posedge "clk"
      moore.event negedge "clk"
    ```
  }];
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
  let description = [{
    The constant operation produces a constant value of a simple bit vector
    type. What is the SBV (Simple Bit Vector) type, it is a zero or
    one-dimensional integer type. For example, `bit`, `logic [0:0]`,
    `reg [31:0]`, or `int`, but `bit[1:0][2:0]`, `int [4:0]`, `bit[5:2]`, or
    `bit []` are not.

    Example:
    ```mlir
      %0 = moore.constant false : !moore.logic
      %1 = moore.constant 0 : !moore.packed<range<bit, 7:0>>
      %3 = moore.constant -7 : !moore.packed<range<bit, 3:0>>
    ```
  }];

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
    
    Example:
    ```mlir
    int a;
    shortint b;
    a = b;  // generates an implicit cast from shortint to int
    ```

    Or explicitly by the user through a type, sign, or const cast expression:
    ```mlir
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
    X, all bits in the result are X.

    See IEEE 1800-2017 § 11.4.3 "Arithmetic operators".

    Example:
    ```mlir
      moore.not %a : !moore.int
      moore.not %b : !moore.bit
    ```
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

    Example:
    ```mlir
      %0 = moore.not %a : !moore.logic
    ```
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

    Example:
    ```mlir
      %0 = moore.bool_cast %in : (!moore.logic) -> (!moore.int)
    ```
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
    result are X.

    See IEEE 1800-2017 § 11.4.3 "Arithmetic operators".

    Example:
    ```mlir
      %0 = moore.add %arg0, %arg1 : !moore.bit
      %c = moore.add %a, %b : !moore.int
    ```
  }];
}

def SubOp : BinaryOpBase<"sub"> {
  let summary = "Subtraction";
  let description = [{
    Subtract the right-hand side from the left-hand side operand. If any bit in
    the two operands is Z or X, all bits in the result are X.

    See IEEE 1800-2017 § 11.4.3 "Arithmetic operators".

    Example:
    ```mlir
      %0 = moore.sub %arg0, %arg1 : !moore.bit
      %c = moore.sub %a, %b : !moore.int
    ```
  }];
}

def MulOp : BinaryOpBase<"mul", [Commutative]> {
  let summary = "Multiplication";
  let description = [{
    Multiply the operands. If any bit in the two operands is Z or X, all bits in
    the result are X.

    See IEEE 1800-2017 § 11.4.3 "Arithmetic operators".

    Example:
    ```mlir
      %0 = moore.mul %arg0, %arg1 : !moore.bit
      %c = moore.mul %a, %b : !moore.int
    ```
  }];
}

def DivOp : BinaryOpBase<"div"> {
  let summary = "Division";
  let description = [{
    Divide the left-hand side by the right-hand side operand. Any fractional
    part is truncated toward zero. If the right-hand side is zero, all bits of
    the result are X. If any bit in the two operands is Z or X, all bits in the
    result are X.

    See IEEE 1800-2017 § 11.4.3 "Arithmetic operators".

    Example:
    ```mlir
      %0 = moore.div %arg0, %arg1 : !moore.bit
      %c = moore.div %a, %b : !moore.integer
    ```
  }];
}

def ModOp : BinaryOpBase<"mod"> {
  let summary = "Remainder";
  let description = [{
    Compute the remainder of the left-hand side divided by the right-hand side
    operand. If the right-hand side is zero, all bits of the result are X. The
    sign of the result is the sign of the left-hand side. If any bit in the two
    operands is Z or X, all bits in the result are X.

    See IEEE 1800-2017 § 11.4.3 "Arithmetic operators".

    Consider the following examples:

    | LHS | RHS | Result |
    |-----|-----|--------|
    |  11 |   3 |      2 |
    | -11 |   3 |     -2 |
    |  11 |  -3 |      2 |
    | -11 |  -3 |     -2 |

    Example:
    ```mlir
      %0 = moore.mod %a, %b : !moore.logic
      %1 = moore.mod %in1, %in2 : !moore.packed<range<logic, 31:0>>
    ```
  }];
}

def AndOp : BinaryOpBase<"and", [Commutative]> {
  let summary = "Bitwise AND operation";
  let description = [{
    Applies the boolean AND operation to each pair of corresponding bits in the
    left- and right-hand side operand. Corresponds to the `&` operator.

    See IEEE 1800-2017 § 11.4.8 "Bitwise operators".

    |   | 0 | 1 | X | Z |
    |---|---|---|---|---|
    | 0 | 0 | 0 | 0 | 0 |
    | 1 | 0 | 1 | X | X |
    | X | 0 | X | X | X |
    | Z | 0 | X | X | X |

    Example:
    ```mlir
      %2 = moore.and %0, %1 : !moore.logic
    ```
  }];
}

def OrOp : BinaryOpBase<"or", [Commutative]> {
  let summary = "Bitwise OR operation";
  let description = [{
    Applies the boolean OR operation to each pair of corresponding bits in the
    left- and right-hand side operand. Corresponds to the `|` operator.

    See IEEE 1800-2017 § 11.4.8 "Bitwise operators".

    |   | 0 | 1 | X | Z |
    |---|---|---|---|---|
    | 0 | 0 | 1 | X | X |
    | 1 | 1 | 1 | 1 | 1 |
    | X | X | 1 | X | X |
    | Z | X | 1 | X | X |

    Example:
    ```mlir
      %c = moore.or %a, %b : !moore.logic
    ```
  }];
}

def XorOp : BinaryOpBase<"xor", [Commutative]> {
  let summary = "Bitwise XOR operation";
  let description = [{
    Applies the boolean XOR operation to each pair of corresponding bits in the
    left- and right-hand side operand. Corresponds to the `^` operator.

    See IEEE 1800-2017 § 11.4.8 "Bitwise operators".

    |   | 0 | 1 | X | Z |
    |---|---|---|---|---|
    | 0 | 0 | 1 | X | X |
    | 1 | 1 | 0 | X | X |
    | X | X | X | X | X |
    | Z | X | X | X | X |

    Example:
    ```mlir
      %c = moore.xor %a, %b : !moore.logic
    ```
  }];
}

class LogicalEqOpBase<string mnemonic> : MooreOp<mnemonic, [
  Pure,
  Commutative,
  SameTypeOperands,
  TypesMatchWith<"result is single bit matching input domain",
    "lhs", "result", [{
    IntType::get($_self.getContext(),
      IntType::getAtomForDomain($_self.cast<UnpackedType>().getDomain()))
  }]>
]> {
  let description = [{
    Compares the bits in the left- and right-hand side operand and returns a
    single bit 0, 1, or X result. If all corresponding bits in the left- and
    right-hand side are equal, and all are 0 or 1 (not X or Z), the two operands
    are considered equal (`eq` returns 1, `ne` returns 0). If any bits are not
    equal, but all are 0 or 1, the two operands are considered not equal (`eq`
    returns 0, `ne` returns 1). If any bit in the two operands is Z or X,
    returns X. `eq` corresponds to the `==` operator and `ne` to the `!=`
    operator.

    See IEEE 1800-2017 § 11.4.5 "Equality operators".
  }];
  let arguments = (ins SimpleBitVectorType:$lhs, SimpleBitVectorType:$rhs);
  let results = (outs SingleBitType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
  }];
}

def EqOp : LogicalEqOpBase<"eq"> { let summary = "Logical equality"; }
def NeOp : LogicalEqOpBase<"ne"> { let summary = "Logical inequality"; }

class CaseEqOpBase<string mnemonic> : MooreOp<mnemonic, [
  Pure,
  Commutative,
  SameTypeOperands
]> {
  let description = [{
    Compares the bits in the left- and right-hand side operand and returns a
    single bit 0 or 1 result. If all corresponding bits in the left- and
    right-hand side are equal (both 0, 1, X, or Z), the two operands are
    considered equal (`case_eq` returns 1, `case_ne` returns 0). If any bits are
    not equal, the two operands are considered not equal (`case_eq` returns 0,
    `case_ne` returns 1). `case_eq` corresponds to the `===` operator and
    `case_ne` to the `!==` operator.

    See IEEE 1800-2017 § 11.4.5 "Equality operators".
  }];
  let arguments = (ins SimpleBitVectorType:$lhs, SimpleBitVectorType:$rhs);
  let results = (outs BitType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs)
  }];
}

def CaseEqOp : CaseEqOpBase<"case_eq"> { let summary = "Case equality"; }
def CaseNeOp : CaseEqOpBase<"case_ne"> { let summary = "Case inequality"; }

class WildcardEqOpBase<string mnemonic> : MooreOp<mnemonic, [
  Pure,
  SameTypeOperands,
  TypesMatchWith<"result is single bit matching input domain",
    "lhs", "result", [{
    IntType::get($_self.getContext(),
      IntType::getAtomForDomain($_self.cast<UnpackedType>().getDomain()))
  }]>
]> {
  let description = [{
    Compares the bits in the left- and right-hand side operand and returns a
    single bit 0, 1, or X result. If any bit in the left-hand side is Z or X,
    returns X. Performs the same comparison as the `eq` and `ne` operations, but
    all right-hand side bits that are X or Z are skipped. Therefore, X and Z in
    the right-hand side act as wildcards or "don't care" values. `wildcard_eq`
    corresponds to the `==?` operator and `wildcard_ne` to the `!=?` operator.

    See IEEE 1800-2017 § 11.4.6 "Wildcard equality operators".
  }];
  let arguments = (ins SimpleBitVectorType:$lhs, SimpleBitVectorType:$rhs);
  let results = (outs SingleBitType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
  }];
}

def WildcardEqOp : WildcardEqOpBase<"wildcard_eq"> {
  let summary = "Wildcard equality";
}
def WildcardNeOp : WildcardEqOpBase<"wildcard_ne"> {
  let summary = "Wildcard inequality";
}

class RelationalOpBase<string mnemonic> : MooreOp<mnemonic, [
  Pure,
  SameTypeOperands,
  TypesMatchWith<"result is single bit matching input domain",
    "lhs", "result", [{
    IntType::get($_self.getContext(),
      IntType::getAtomForDomain($_self.cast<UnpackedType>().getDomain()))
  }]>
]> {
  let description = [{
    Compares the left- and right-hand side operand and returns a single bit 0,
    1, or X result. If any bit in the two operands is Z or X, returns X.
    Otherwise, if all bits are 0 or 1, `lt`, `le`, `gt`, and `ge` return whether
    the left-hand side is less than, less than or equal to, greater than, or
    greater than or equal to the right-hand side, respectively. `lt` corresponds
    to the `<` operator, `le` to `<=`, `gt` to `>`, and `ge` to `>=`.

    See IEEE 1800-2017 § 11.4.4 "Relational operators".
  }];
  let arguments = (ins SimpleBitVectorType:$lhs, SimpleBitVectorType:$rhs);
  let results = (outs SingleBitType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)
  }];
}

def LtOp : RelationalOpBase<"lt"> { let summary = "Less than"; }
def LeOp : RelationalOpBase<"le"> { let summary = "Less than or equal"; }
def GtOp : RelationalOpBase<"gt"> { let summary = "Greater than"; }
def GeOp : RelationalOpBase<"ge"> { let summary = "Greater than or equal"; }

#endif // CIRCT_DIALECT_MOORE_MOOREOPS
