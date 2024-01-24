import functools
import operator
from tensor import Tensor
from ops import MemoryOp, BinaryOp

class Program:
    def __init__(self) -> None: #Have to have indentation tracker
        self.headers: list[str] = ["#include <stdlib.h>"]
        self.main_start: list[str] = ["int main() {"]
        self.main_body: list[str] = []
        self.main_end: list[str] = ["}"]

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
    



