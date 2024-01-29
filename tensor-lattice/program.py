import functools
import operator
from tensor import Tensor
from ops import MemoryOp, BinaryOp
from ir import IR, to_ir

class Program:
    def __init__(self, child: Tensor) -> None: #Have to have indentation tracker
        self.child: Tensor = child
        self.indent_level: int = 0
        self.headers: list[str] = ["#include <stdlib.h>\n"]
        self.main_start: str = ""
        self.main_body: list[str] = []
        self.main_end: list[str] = ["}"]

    def generate(self) -> list[str]:
        kernel_name: str = create_name_and_verify_data_type(child = self.child)

        self.main_start = f"{self.child._memory._data_type} {kernel_name}("

        kernel: list[IR] = to_ir(end = self.child)

        for i in kernel:
            if i.op == "ARG":
                self.main_start += f"{i.data_type} {i.value}"
                if "out" in i.value:
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
                self.main_body.append(gen_store(ir_store = i, indent_count = self.indent_level))
            elif i.op == "PLEQ":
                self.main_body.append(gen_pleq(ir_store = i, indent_count = self.indent_level))
            
        return "".join(self.headers + [self.main_start] + self.main_body + self.main_end)

def create_name_and_verify_data_type(child: Tensor) -> str:
    start_buffer: Tensor = child._topological_sort()[0]
    if child._memory._data_type != start_buffer._memory._data_type:
        raise ValueError("The data types of the input and output buffer do not match.")
    name: str = "_".join(["k"] + [str(i) for i in start_buffer._memory.view] + ["to"] + [str(i) for i in child._memory.view])
    return name

# Some of this gets redirected to the gen_load
def gen_store(ir_store: IR, indent_count: int) -> str:
    map_ops: dict[str, str] = {"ADD": "+", "DIV": "/", "MUL": "*", "SUB": "-"}
    left_side = ("\t" * indent_count) + gen_load(ir_load = ir_store.dependencies[0]) + " = "
    right_side = gen_load(ir_load = ir_store.dependencies[1].dependencies[0]) + f" {map_ops[ir_store.dependencies[1].op]} " + gen_load(ir_load = ir_store.dependencies[1].dependencies[1]) + ";\n"
    store = left_side + right_side
    return store

def gen_pleq(ir_store: IR, indent_count: int) -> str:
    left_side = ("\t" * indent_count) + gen_load(ir_load = ir_store.dependencies[0]) + " += "
    right_side = gen_load(ir_load = ir_store.dependencies[1]) + ";\n"
    pleq = left_side + right_side
    return pleq

def gen_load(ir_load: IR) -> str:
    return f"(*({ir_load.value} + " + gen_tensor_indices(load_op = ir_load.dependencies[0]) + "))"

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

# Take care of the indent in render program instead?
def render_load_op(indent_number: int, data_type:str, shape: list[int], tensor_number: int) -> str:
    buffer = (indent_number * "\t") + data_type + "* "
    buffer += "tensor_" + str(tensor_number)
    buffer += " = (" + data_type + "*) "
    buffer += "malloc(" + str(functools.reduce(operator.mul, shape))
    buffer += " * sizeof(" + data_type
    buffer += "));"
    return buffer # Can use f-strings here to make it shorter? Just assign to variables first

def render_tensor_indices(name: str, index_bounds: dict[str, int]):
    tensor = f"(*({name}"
    for index in index_bounds:
        if index_bounds[index] > 0:
            tensor += f"+({index}*{index_bounds[index]})"
        elif index_bounds[index] == 0:
            tensor += f"+({index})"
    tensor += "))"
    return tensor

# Fix stride for each Tensor, not just one
def render_binary_op(tensor_1: str, tensor_2: str, store: str, shape: list[int], stride: list[int], op: str, indent: int):
    binary_op = []
    loop_buffer = []
    index_bounds = {}
    indent_count = indent
    iter_count = 0
    for num, i in enumerate(shape):
        if i != 1:
            binary_op.append(("\t" * indent) + f"int iter_{iter_count};")
            index_bounds[f"iter_{iter_count}"] = stride[num]
            iter_count += 1
            temp_loop = ("\t" * indent_count) + f"for (iter_{iter_count}=0; iter_{iter_count}<{i}; iter_{iter_count}++) " + "{"
            loop_buffer.append(temp_loop)
            indent_count += 1
    binary_op = binary_op + loop_buffer
    expression = ("\t" * indent_count) + render_tensor_indices(name = store, index_bounds = index_bounds)
    expression += " = " + render_tensor_indices(name = tensor_1, index_bounds = index_bounds)
    expression += " " + op + " " + render_tensor_indices(name = tensor_2, index_bounds = index_bounds) + ";"
    binary_op.append(expression)
    for i in range(indent_count - 1, indent - 1, -1):
        binary_op.append(("\t" * i) + "}")
    return binary_op


def render_program(end: Tensor) -> Program:
    program = Program()
    indent_tracker = 1
    data_tracker = 0
    tensor_schedule = end._topological_sort()
    tensor_names = {}
    for i in tensor_schedule:
        tensor_names[i] = f"tensor_{data_tracker}"
        if isinstance(i._op, MemoryOp):
            program.main_body.append(render_load_op(indent_number = indent_tracker, 
                                                    data_type = i._memory._data_type, 
                                                    shape = i._memory.view, 
                                                    tensor_number = data_tracker))
        elif isinstance(i._op, BinaryOp):
            pass
        data_tracker += 1
    



