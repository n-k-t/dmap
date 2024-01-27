from __future__ import annotations
from ops import BinaryOp, MemoryOp, MovementOp
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
            open_loop_to_ir(lower_bound = 0, upper_bound = i, name = f"index_{num}", kernel = kernel, consts = consts)
            axis_loops[num] = kernel[-1]

    store_add: list[IR] = []
    store_load: list[IR] = []

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
                temp_op: IR = IR(op = "ADD", data_type = "int", value = "", dependencies = [store_add[-1], temp_op])
                kernel.append(temp_op)
            store_add.append(temp_op)
        temp_load: IR = IR(op = "LOAD", data_type = "float", value = tensors[tensor].value, dependencies = [kernel[-1]])
        store_load.append(temp_load)
        kernel.append(temp_load)

    op_shortening: dict[str, str] = {"ADDITION": "ADD", "DIVISION": "DIV", \
                                        "MULTIPLICATION": "MUL", "SUBTRACTION": "SUB"}

    temp_comb_op: IR = IR(op = op_shortening[child._op.op], data_type = "float", value = "", dependencies = store_load)
    kernel.append(temp_comb_op)

    store_add: list[IR] = []
    store_load: list[IR] = []

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
            temp_op: IR = IR(op = "ADD", data_type = "int", value = "", dependencies = [store_add[-1], temp_op])
            kernel.append(temp_op)
        store_add.append(temp_op)
    temp_load: IR = IR(op = "LOAD", data_type = "float", value = tensors[child].value, dependencies = [kernel[-1]])
    store_load.append(temp_load)
    kernel.append(temp_load)

    temp_store: IR = IR(op = "STORE", data_type = "float", value = store_load[0].value, dependencies = [store_load[0], temp_comb_op])
    kernel.append(temp_store)

    for loop in axis_loops.values():
        close_loop_to_ir(loop = loop, kernel = kernel)


def open_loop_to_ir(lower_bound: int, upper_bound: int, name: str, kernel: list[IR], consts: dict[int | float, IR]) -> None:
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

def close_loop_to_ir(loop: IR, kernel: list[IR]) -> None:
    loop_set: set[IR] = set([loop])
    delta_len: int = len(loop_set)
    while delta_len != 0:
        start_len: int = len(loop_set)
        for ir in kernel:
            if loop_set.intersection([i for i in ir.dependencies]):
                loop_set.add(ir)
        delta_len = len(loop_set) - start_len
    insert_index: int = max([kernel.index(i) for i in loop_set]) + 1
    temp_end: IR = IR(op = "END", data_type = "", value = "", dependencies = [loop])
    kernel.insert(insert_index, temp_end)

def to_ir(end: Tensor) -> list[IR]:
    kernel_ir: list[IR] = []

    tensor_list: list[Tensor] = end._topological_sort()
    op_tensors: dict[Tensor, BinaryOp] = {}
    tensor_pointers: dict[Tensor, IR] = {}
    const_pointers: dict[int | float, IR] = {}

    for num, i in enumerate(tensor_list):
        if isinstance(i._op, MemoryOp):
            if num == len(tensor_list) - 1:
                temp: IR = IR(op = "ARG", data_type = "float*", value = "tensor_out", dependencies = [])
                kernel_ir.append(temp)
                tensor_pointers[i] = temp
            else:
                temp: IR = IR(op = "ARG", data_type = "float*", value = f"tensor_{num}", dependencies = [])
                kernel_ir.append(temp)
                tensor_pointers[i] = temp
        elif isinstance(i._op, BinaryOp):
            if num == len(tensor_list) - 1:
                temp: IR = IR(op = "ARG", data_type = "float*", value = "tensor_out", dependencies = [])
                kernel_ir.append(temp)
                tensor_pointers[i] = temp
            else:
                temp: IR = IR(op = "ARG", data_type = "float*", value = f"tensor_{num}", dependencies = [])
                kernel_ir.append(temp)
                tensor_pointers[i] = temp
            op_tensors[i] = i._op
        elif isinstance(i._op, MovementOp):
            # TODO: Reshapes aren't working because the binary operation still thinks that it is using the original
            # tensor. I need to update to this so that the correct tensor is considered.
            # if i in op_tensors:
            #     op_tensors[i] = tensor_pointers[i._parents[0]]
            #     del op_tensors[i._parents[0]]
            tensor_pointers[i] = tensor_pointers[i._parents[0]]
            del tensor_pointers[i._parents[0]]

    for child in op_tensors:
        bin_op_to_ir(child = child, kernel = kernel_ir, tensors = tensor_pointers, consts = const_pointers)
    
    return kernel_ir