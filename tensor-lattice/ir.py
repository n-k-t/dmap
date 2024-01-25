from __future__ import annotations
from ops import BinaryOp, MemoryOp
from tensor import Tensor

class IR:
    def __init__(
            self,
            op: str,
            data_type: str,
            value: int | float | None = None,
            dependencies: list[IR] = []
        ) -> None:
        self.op: str = op
        self.data_type: str = data_type
        self.value: int | float | None = value
        self.dependencies: list[IR] = dependencies

def bin_op_to_ir(child: Tensor, kernel: list[IR], tensors: dict[Tensor, IR], consts: dict[int | float, IR]) -> None:
    for i in child._parents[0]._memory.view:
        if i != 1:
            loop_to_ir(lower_bound = 0, upper_bound = i, kernel = kernel, consts = consts)

def loop_to_ir(lower_bound: int, upper_bound: int, kernel: list[IR], consts: dict[int | float, IR]) -> None:
    if lower_bound not in consts:
        temp_bound: IR = IR(op = "CONST", data_type = "int", value = lower_bound, dependencies = [])
        consts[0] = temp_bound
        kernel.append(temp_bound)
    if upper_bound not in consts:
        temp_bound: IR = IR(op = "CONST", data_type = "int", value = upper_bound, dependencies = [])
        consts[0] = temp_bound
        kernel.append(temp_bound)
    temp_loop: IR = IR(op = "FOR", data_type = "", value = "", dependencies = [consts[lower_bound], consts[upper_bound]])
    kernel.append(temp_loop)

def to_ir(end: Tensor) -> list[IR]:
    kernel_ir: list[IR] = []

    tensor_list: list[Tensor] = end._topological_sort()
    op_tensors: dict[Tensor, BinaryOp] = {}
    tensor_pointers: dict[Tensor, IR] = {}
    const_pointers: dict[int | float, IR] = {}

    for num, i in enumerate(tensor_list):
        if isinstance(i._op, MemoryOp):
            temp: IR = IR(op = "ARG", data_type = "float*", value = f"tensor_{num}", dependencies = [])
            kernel_ir.append(temp)
            tensor_pointers[i] = temp
        elif isinstance(i._op, BinaryOp):
            op_tensors[i] = i._op
        if num == len(tensor_list) - 1:
            temp: IR = IR(op = "ARG", data_type = "float*", value = f"tensor_out", dependencies = [])
            kernel_ir.append(temp)
            tensor_pointers[i] = temp

    for child in op_tensors:
        bin_op_to_ir(child = child, kernel = kernel_ir, tensors = tensor_pointers, consts = const_pointers)
    
    return kernel_ir