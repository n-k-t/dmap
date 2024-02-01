from tensor import Tensor
from ops import MemoryOp, BinaryOp, ReduceOp
from first_pass import IR, preliminary_ir, separate_kernels
from c_ir import to_c_ir

class Program:
    def __init__(self, head: Tensor) -> None:
        self.ast_slices: list[Tensor] = separate_kernels(end = head)
        self.names: list[str] = []
        for tensor in self.ast_slices:
            if isinstance(tensor._op, ReduceOp):
                self.names.append("_".join(["re"] + [str(i) for i in tensor._parents[0]._memory.view] + ["to"] + [str(i) for i in tensor._memory.view]))
            else:
                self.names.append("_".join(["el"] + [str(i) for i in tensor._parents[0]._memory.view] + ["to"] + [str(i) for i in tensor._memory.view]))
        self.code: list[str] = []
        for tensor, name in zip(self.ast_slices, self.names):
            temp: Code = Code(tensor, name)
            self.code.append(temp.generate())


class Code:
    def __init__(self, child: Tensor, name: str) -> None:
        self.child: Tensor = child
        self.name: str = name
        self.indent_level: int = 0
        self.headers: list[str] = ["#include <stdlib.h>\n"]
        self.main_start: str = ""
        self.main_body: list[str] = []
        self.main_end: list[str] = ["}"]

    def generate(self) -> list[str]:
        kernel_name: str = self.name

        self.main_start = f"{self.child._memory._data_type} {kernel_name}("

        kernel: list[IR] = preliminary_ir(ast_slice = self.child)

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
                self.main_body.append(("\t" * self.indent_level) + f"for (int {i.value} = {i.dependencies[0].value}; {i.value} < {i.dependencies[1].value}; {i.value}++)" + "{\n")
                self.indent_level += 1
            elif i.op == "END":
                self.main_body.append(("\t" * (self.indent_level - 1)) + "}\n")
                self.indent_level -= 1
            elif i.op == "STORE":
                if isinstance(self.child._op, ReduceOp):
                    self.main_body.append(gen_store(ir_store = i, indent_count = self.indent_level, reduce = True))
                else:
                    self.main_body.append(gen_store(ir_store = i, indent_count = self.indent_level, reduce = False))
            
        return "".join(self.headers + [self.main_start] + self.main_body + self.main_end)

def gen_store(ir_store: IR, indent_count: int, reduce: bool) -> str:
    if reduce:
        left_side = ("\t" * indent_count) + gen_load(ir_load = ir_store.dependencies[0]) + " += "
        right_side = gen_load(ir_load = ir_store.dependencies[1].dependencies[0]) + ";\n"
        pleq = left_side + right_side
        return pleq
    else:
        map_ops: dict[str, str] = {"ADD": "+", "DIV": "/", "MUL": "*", "SUB": "-"}
        left_side = ("\t" * indent_count) + gen_load(ir_load = ir_store.dependencies[0]) + " = "
        right_side = gen_load(ir_load = ir_store.dependencies[1].dependencies[0]) + f" {map_ops[ir_store.dependencies[1].op]} " + gen_load(ir_load = ir_store.dependencies[1].dependencies[1]) + ";\n"
        store = left_side + right_side
        return store

def gen_load(ir_load: IR) -> str:
    return f"(*({ir_load.value} + " + gen_tensor_indices(load_op = ir_load.dependencies[1]) + "))"

def gen_tensor_indices(load_op: IR) -> str:
    map_ops: dict[str, str] = {"ADD": "+", "DIV": "/", "MUL": "*", "SUB": "-"}
    int_ops: list[str] = ["ADD", "DIV", "MUL", "SUB"]
    if load_op.dependencies[0].op in int_ops:
        left_expression = gen_tensor_indices(load_op = load_op.dependencies[0])
    else:
        left_expression = str(load_op.dependencies[0].value)
    if load_op.dependencies[1].op in int_ops:
        right_expression = gen_tensor_indices(load_op = load_op.dependencies[1])
    else:
        right_expression = str(load_op.dependencies[1].value)
    expression = f"({left_expression} {map_ops[load_op.op]} {right_expression})"
    return expression