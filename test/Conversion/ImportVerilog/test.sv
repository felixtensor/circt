// RUN: circt-translate --import-verilog %s | FileCheck %s

// CHECK-LABEL: moore.module @Port
// CHECK-NEXT: %a = moore.port In : !moore.packed<range<logic, 3:0>>
// CHECK-NEXT: %b = moore.port Out : !moore.packed<range<logic, 3:0>>
module Port(input [3:0] a, output [3:0] b);
endmodule
