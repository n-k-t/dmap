from dmap.tensor import Tensor
from dmap.ops import ReduceOp
from dmap.first_pass import IR
from dmap.c_ir import to_c_ir


class Code:
    def __init__(self, child: Tensor, ir: IR) -> None:
        self.child: Tensor = child
        self.name: str = ""
        if isinstance(child._op, ReduceOp):
            self.name = "_".join(["R"] + [str(i) for i in child._parents[0]._memory.view] + ["to"] + [str(i) for i in child._memory.view])
        else:
            self.name = "_".join(["E"] + [str(i) for i in child._parents[0]._memory.view] + ["to"] + [str(i) for i in child._memory.view])
        self.indent_level: int = 0
        self.headers: list[str] = ["#include <stdlib.h>\n"]
        self.main_start: str = ""
        self.main_body: list[str] = []
        self.main_end: list[str] = ["}"]
        self.code: str = self.generate(ir)

    def generate(self, kernel: list[IR]) -> list[str]:
        kernel_name: str = self.name

        self.main_start = f"{self.child._memory._data_type} {kernel_name}("

        c_kernel: list[IR] = to_c_ir(kernel = kernel)

        for i in c_kernel:
            if i.op == "ARG":
                self.main_start += f"{i.data_type} {i.value}"
                if "result" in i.value:
                    self.main_start += ") {\n"
                    self.indent_level += 1
                else:
                    self.main_start += ", "
            elif i.op == "FOR":
                self.main_body.append(("\t" * self.indent_level) + f"for (int {i.value} = {i.dependencies[0].value}; {i.value} < {i.dependencies[1].value}; {i.value}++)" + " {\n")
                self.indent_level += 1
            elif i.op == "END":
                self.indent_level -= 1
                self.main_body.append(("\t" * (self.indent_level)) + "}\n")
            elif (i.op == "STORE") and ("phi" not in i.value):
                if isinstance(self.child._op, ReduceOp):
                    self.main_body.append(self.gen_store(i, self.indent_level, True))
                else:
                    self.main_body.append(self.gen_store(i, self.indent_level, False))
            elif i.op == "IF/ELSE":
                if i.dependencies[0].op == "CMPR":
                    self.main_body.append(("\t" * self.indent_level) + f"if {self.gen_cmpr(i.dependencies[0])}" + " {\n")
                elif i.dependencies[0].op == "OR":
                    self.main_body.append(("\t" * self.indent_level) + f"if ({self.gen_or(i.dependencies[0])})" + " {\n")
                self.indent_level += 1
                self.main_body.append(("\t" * self.indent_level) + f"{i.dependencies[1].value} = {self.gen_load(i.dependencies[1].dependencies[1])};\n")
                self.indent_level -= 1
                self.main_body.append(("\t" * self.indent_level) + "} else {\n")
                self.indent_level += 1
                self.main_body.append(("\t" * self.indent_level) + f"{i.dependencies[2].value} = {float(i.dependencies[2].dependencies[1].value)};\n")
                self.indent_level -= 1
                self.main_body.append(("\t" * self.indent_level) + "}\n")
            
        return "".join(self.headers + [self.main_start] + self.main_body + self.main_end)

    def gen_store(self, ir_store: IR, indent_count: int, reduce: bool) -> str:
        if reduce:
            left_side = ("\t" * indent_count) + self.gen_load(ir_store.dependencies[0]) + " += "
            right_side = self.gen_load(ir_store.dependencies[1].dependencies[0]) + ";\n"
            pleq = left_side + right_side
            return pleq
        else:
            map_ops: dict[str, str] = {"ADD": "+", "DIV": "/", "MUL": "*", "SUB": "-"}
            left_side = ("\t" * indent_count) + self.gen_load(ir_store.dependencies[0]) + " = "
            right_side = self.gen_load(ir_store.dependencies[1].dependencies[0]) + f" {map_ops[ir_store.dependencies[1].op]} " + self.gen_load(ir_store.dependencies[1].dependencies[1]) + ";\n"
            store = left_side + right_side
            return store

    def gen_load(self, ir_load: IR) -> str:
        if ir_load.op == "PHI":
            return f"{ir_load.value}"
        else:
            return f"(*({ir_load.value} + " + self.gen_tensor_indices(ir_load.dependencies[1]) + "))"

    def gen_tensor_indices(self, load_op: IR) -> str:
        map_ops: dict[str, str] = {"ADD": "+", "DIV": "/", "MUL": "*", "SUB": "-"}
        int_ops: list[str] = ["ADD", "DIV", "MUL", "SUB"]
        if load_op.dependencies[0].op in int_ops:
            left_expression = self.gen_tensor_indices(load_op.dependencies[0])
        else:
            left_expression = str(load_op.dependencies[0].value)
        if load_op.dependencies[1].op in int_ops:
            right_expression = self.gen_tensor_indices(load_op.dependencies[1])
        else:
            right_expression = str(load_op.dependencies[1].value)
        expression = f"({left_expression} {map_ops[load_op.op]} {right_expression})"
        return expression

    def gen_cmpr(self, ir_cmpr: IR) -> str:
        return f"({ir_cmpr.dependencies[0].value} {ir_cmpr.value} {ir_cmpr.dependencies[1].value})"

    def gen_or(self, ir_or: IR) -> str:
        if ir_or.dependencies[0].op == "OR":
            left_side = self.gen_or(ir_or.dependencies[0])
        else:
            left_side = self.gen_cmpr(ir_or.dependencies[0])
        right_side = self.gen_cmpr(ir_or.dependencies[1])
        return f"{left_side} || {right_side}"