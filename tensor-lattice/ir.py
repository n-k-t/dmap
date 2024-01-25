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

    def __repr__(self) -> str:
        return f"OP: {self.op:>10},    DT: {self.data_type:>10},    VAL: {str(self.value):>10},    DEP: {[(i.op, i.value) for i in self.dependencies]}"

def bin_op_to_ir(child: Tensor, kernel: list[IR], tensors: dict[Tensor, IR], consts: dict[int | float, IR]) -> None:
    axis_loops: dict[int, IR] = {}

    for num, i in enumerate(child._parents[0]._memory.view):
        if i != 1:
            loop_to_ir(lower_bound = 0, upper_bound = i, name = f"index_{num}", kernel = kernel, consts = consts)
            axis_loops[num] = kernel[-1]

    store_add: IR | None = None

    for tensor in child._parents:
        for index, axis in enumerate(axis_loops):
            if axis not in consts:
                temp_bound: IR = IR(op = "CONST", data_type = "int", value = axis, dependencies = [])
                consts[axis] = temp_bound
                kernel.append(temp_bound)
            if tensor._memory.stride[index] not in consts:
                temp_bound: IR = IR(op = "CONST", data_type = "int", value = tensor._memory.stride[index], dependencies = [])
                consts[tensor._memory.stride[index]] = temp_bound
                kernel.append(temp_bound)
            temp_op: IR = IR(op = "MUL", data_type = "int", value = "", dependencies = [axis_loops[axis], consts[tensor._memory.stride[index]]])
            kernel.append(temp_op)
            if index != 0:
                temp_op: IR = IR(op = "ADD", data_type = "int", value = "", dependencies = [store_add, temp_op])
                kernel.append(temp_op)
            store_add = temp_op
        temp_load: IR = IR(op = "LOAD", data_type = "float", value = "", dependencies = [tensors[tensor], kernel[-1]])
        kernel.append(temp_load)

def loop_to_ir(lower_bound: int, upper_bound: int, name: str, kernel: list[IR], consts: dict[int | float, IR]) -> None:
    if lower_bound not in consts:
        temp_bound: IR = IR(op = "CONST", data_type = "int", value = lower_bound, dependencies = [])
        consts[lower_bound] = temp_bound
        kernel.append(temp_bound)
    if upper_bound not in consts:
        temp_bound: IR = IR(op = "CONST", data_type = "int", value = upper_bound, dependencies = [])
        consts[upper_bound] = temp_bound
        kernel.append(temp_bound)
    temp_loop: IR = IR(op = "FOR", data_type = "", value = name, dependencies = [consts[lower_bound], consts[upper_bound]])
    kernel.append(temp_loop)

def to_ir(end: Tensor) -> list[IR]:
    kernel_ir: list[IR] = []

    tensor_list: list[Tensor] = end._topological_sort()
    op_tensors: dict[Tensor, BinaryOp] = {}
    tensor_pointers: dict[Tensor, IR] = {}
    const_pointers: dict[int | float, IR] = {}

    for num, i in enumerate(tensor_list):
        if isinstance(i._op, MemoryOp):
            if num == len(tensor_list) - 1:
                temp: IR = IR(op = "ARG", data_type = "float*", value = f"tensor_out", dependencies = [])
                kernel_ir.append(temp)
                tensor_pointers[i] = temp
            else:
                temp: IR = IR(op = "ARG", data_type = "float*", value = f"tensor_{num}", dependencies = [])
                kernel_ir.append(temp)
                tensor_pointers[i] = temp
        elif isinstance(i._op, BinaryOp):
            if num == len(tensor_list) - 1:
                temp: IR = IR(op = "ARG", data_type = "float*", value = f"tensor_out", dependencies = [])
                kernel_ir.append(temp)
                tensor_pointers[i] = temp
            op_tensors[i] = i._op

    for child in op_tensors:
        bin_op_to_ir(child = child, kernel = kernel_ir, tensors = tensor_pointers, consts = const_pointers)
    
    return kernel_ir