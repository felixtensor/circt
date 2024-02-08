//===- MIRStatements.td - Moore MIR statements ops ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the ops for Moore MIR statements.
//
//===----------------------------------------------------------------------===//

include "circt/Dialect/Moore/MooreTypes.td"

//===----------------------------------------------------------------------===//
// Assignment Statements
//===----------------------------------------------------------------------===//

def CAssignOp : MIROp<"cassign", [SameTypeOperands]> {
  let summary = "Continuous assignment";
  let description = [{
    A SystemVerilog assignment statement 'x = y;'.
    These occur in module scope.  See SV Spec 10.3.2.

    Example:
    ```mlir
      ......
      %z = moore.port Out : !moore.logic
      ......
      %1 = moore.not %0 : !moore.logic
      moore.mir.cassign %z, %1 : !moore.logic
    ```
  }];

  let arguments = (ins AnyType:$dest, AnyType:$src);
  let results = (outs);
  let assemblyFormat = [{
    $dest `,` $src  attr-dict `:` qualified(type($src))
  }];
}

def BPAssignOp : MIROp<"bpassign", [SameTypeOperands]> {
  let summary = "Blocking procedural assignment";
  let description = [{
    A SystemVerilog blocking procedural assignment statement 'x = y;'.
    These occur in a sequential block.  See SV Spec 10.4.1.

    Example:
    ```mlir
      ......
      %0 = moore.variable : !moore.reg
      moore.procedure always {
        moore.event posedge "clk"
        ......  
        moore.mir.bpassign %0, %2 : !moore.reg
      }
    ```
  }];

  let arguments = (ins AnyType:$dest, AnyType:$src);
  let results = (outs);
  let assemblyFormat = [{
    $dest `,` $src  attr-dict `:` qualified(type($src))
  }];
}

def PAssignOp : MIROp<"passign", [SameTypeOperands]> {
  let summary = "Nonblocking procedural assignment";
  let description = [{
    A SystemVerilog nonblocking procedural assignment statement 'x <= y;'.
    These occur in a sequential block.  See SV Spec 10.4.12.
    
    Example:
    ```mlir
      ......
      %0 = moore.variable : !moore.reg
      moore.procedure always {
        moore.event posedge "clk"
        ......  
        moore.mir.passign %0, %2 : !moore.reg
      }
    ```
  }];

  let arguments = (ins AnyType:$dest, AnyType:$src);
  let results = (outs);
  let assemblyFormat = [{
    $dest `,` $src  attr-dict `:` qualified(type($src))
  }];
}

def PCAssignOp : MIROp<"pcassign", [SameTypeOperands]> {
  let summary = "Procedural continuous assignment";
  let description = [{
    A SystemVerilog assignment statement 'assign x = y;'.
    The procedural continuous assignments (using keywords assign and force) are
    procedural statements that allow expressions to be driven continuously onto
    variables or nets These occur in procedural scope.  See SV Spec 10.6

    Example:
    ```mlir
      moore.mir.pcassign %a, %b : !moore.int
    ```
  }];
  
  let arguments = (ins AnyType:$dest, AnyType:$src);
  let results = (outs);
  let assemblyFormat = [{
    $dest `,` $src  attr-dict `:` qualified(type($src))
  }];
}
