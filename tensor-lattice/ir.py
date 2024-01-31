from __future__ import annotations
from copy import deepcopy
from ops import BinaryOp, MemoryOp, MovementOp, ReduceOp
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
                kernel.append(make_ir(op = "CONST", data_type = "int", value = axis, dependencies = []))
                consts[axis] = kernel[-1]
            if tensor._memory.stride[axis] not in consts:
                kernel.append(make_ir(op = "CONST", data_type = "int", value = tensor._memory.stride[axis], dependencies = []))
                consts[tensor._memory.stride[axis]] = kernel[-1]
            temp_op: IR = make_ir(op = "MUL", data_type = "int", value = "", dependencies = [axis_loops[axis], consts[tensor._memory.stride[axis]]])
            kernel.append(temp_op)
            if index != 0:
                temp_op: IR = make_ir(op = "ADD", data_type = "int", value = "", dependencies = [store_add[-1], temp_op])
                kernel.append(temp_op)
            store_add.append(temp_op)
        temp_load: IR = make_ir(op = "LOAD", data_type = "float", value = tensors[tensor].value, dependencies = [kernel[-1]])
        store_load.append(temp_load)
        kernel.append(temp_load)

    op_shortening: dict[str, str] = {"ADDITION": "ADD", "DIVISION": "DIV", \
                                        "MULTIPLICATION": "MUL", "SUBTRACTION": "SUB"}

    temp_comb_op: IR = make_ir(op = op_shortening[child._op.op], data_type = "float", value = "", dependencies = store_load)
    kernel.append(temp_comb_op)

    store_add: list[IR] = []
    store_load: list[IR] = []

    for index, axis in enumerate(axis_loops):
        if axis not in consts:
            kernel.append(make_ir(op = "CONST", data_type = "int", value = axis, dependencies = []))
            consts[axis] = kernel[-1]
        if child._memory.stride[axis] not in consts:
            kernel.append(make_ir(op = "CONST", data_type = "int", value = child._memory.stride[axis], dependencies = []))
            consts[child._memory.stride[axis]] = kernel[-1]
        temp_op: IR = make_ir(op = "MUL", data_type = "int", value = "", dependencies = [axis_loops[axis], consts[child._memory.stride[axis]]])
        kernel.append(temp_op)
        if index != 0:
            temp_op: IR = make_ir(op = "ADD", data_type = "int", value = "", dependencies = [store_add[-1], temp_op])
            kernel.append(temp_op)
        store_add.append(temp_op)
    kernel.append(make_ir(op = "LOAD", data_type = "float", value = tensors[child].value, dependencies = [kernel[-1]]))

    kernel.append(make_ir(op = "STORE", data_type = "float", value = kernel[-1].value, dependencies = [kernel[-1], temp_comb_op]))

    for loop in axis_loops.values():
        close_loop_to_ir(loop = loop, kernel = kernel)

def red_op_to_ir(child: Tensor, kernel: list[IR], tensors: dict[Tensor, IR], consts: dict[int | float, IR]) -> None:
    reduce_axis: int = child._op.axis
    axis_loops: dict[int, IR] = {}

    for num, i in enumerate(child._parents[0]._memory.view):
        if i != 1:
            if num != reduce_axis:
                open_loop_to_ir(lower_bound = 0, upper_bound = i, name = f"index_{num}", kernel = kernel, consts = consts)
                axis_loops[num] = kernel[-1]
    
    open_loop_to_ir(lower_bound = 0, upper_bound = child._parents[0]._memory.view[reduce_axis], name = f"index_{reduce_axis}", kernel = kernel, consts = consts)
    axis_loops[reduce_axis] = kernel[-1]

    store_add: list[IR] = []

    for index, axis in enumerate(axis_loops):
        if axis not in consts:
            kernel.append(make_ir(op = "CONST", data_type = "int", value = axis, dependencies = []))
            consts[axis] = kernel[-1]
        if child._parents[0]._memory.stride[axis] not in consts:
            kernel.append(make_ir(op = "CONST", data_type = "int", value = child._parents[0]._memory.stride[axis], dependencies = []))
            consts[child._parents[0]._memory.stride[axis]] = kernel[-1]
        temp_op: IR = make_ir(op = "MUL", data_type = "int", value = "", dependencies = [axis_loops[axis], consts[child._parents[0]._memory.stride[axis]]])
        kernel.append(temp_op)
        if index != 0:
            temp_op: IR = make_ir(op = "ADD", data_type = "int", value = "", dependencies = [store_add[-1], temp_op])
            kernel.append(temp_op)
        store_add.append(temp_op)
    kernel.append(make_ir(op = "LOAD", data_type = "float", value = tensors[child._parents[0]].value, dependencies = [kernel[-1]]))

    temp_pleq: IR = kernel[-1]

    # When working on loops, remove instances that multiply by 0.
    temp_red_stride: list[int] = deepcopy(child._memory.stride)
    temp_red_stride.insert(reduce_axis, 0)

    store_add: list[IR] = []

    for index, axis in enumerate(axis_loops):
        if axis not in consts:
            kernel.append(make_ir(op = "CONST", data_type = "int", value = axis, dependencies = []))
            consts[axis] = kernel[-1]
        if temp_red_stride[axis] not in consts:
            kernel.append(make_ir(op = "CONST", data_type = "int", value = temp_red_stride[axis], dependencies = []))
            consts[temp_red_stride[axis]] = kernel[-1]
        temp_op: IR = make_ir(op = "MUL", data_type = "int", value = "", dependencies = [axis_loops[axis], consts[temp_red_stride[axis]]])
        kernel.append(temp_op)
        if index != 0:
            temp_op: IR = make_ir(op = "ADD", data_type = "int", value = "", dependencies = [store_add[-1], temp_op])
            kernel.append(temp_op)
        store_add.append(temp_op)
    kernel.append(make_ir(op = "LOAD", data_type = "float", value = tensors[child].value, dependencies = [kernel[-1]]))

    # This currently only supports the SUM reduce operation.
    kernel.append(make_ir(op = "PLEQ", data_type = "float", value = kernel[-1].value, dependencies = [kernel[-1], temp_pleq]))

    for loop in axis_loops.values():
        close_loop_to_ir(loop = loop, kernel = kernel)

def make_ir(op: str, data_type: str, value: int | float | None, dependencies = list[IR]) -> IR:
    return IR(op = op, data_type = data_type, value = value, dependencies = dependencies)

def open_loop_to_ir(lower_bound: int, upper_bound: int, name: str, kernel: list[IR], consts: dict[int | float, IR]) -> None:
    if lower_bound not in consts:
        kernel.append(make_ir(op = "CONST", data_type = "int", value = lower_bound, dependencies = []))
        consts[lower_bound] = kernel[-1]
    if upper_bound not in consts:
        kernel.append(make_ir(op = "CONST", data_type = "int", value = upper_bound, dependencies = []))
        consts[upper_bound] = kernel[-1]
    kernel.append(make_ir(op = "FOR", data_type = "", value = name, dependencies = [consts[lower_bound], consts[upper_bound]]))

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
                kernel_ir.append(make_ir(op = "ARG", data_type = "float*", value = "tensor_out", dependencies = []))
                tensor_pointers[i] = kernel_ir[-1]
            else:
                kernel_ir.append(make_ir(op = "ARG", data_type = "float*", value = f"tensor_{num}", dependencies = []))
                tensor_pointers[i] = kernel_ir[-1]
        elif isinstance(i._op, BinaryOp):
            if num == len(tensor_list) - 1:
                kernel_ir.append(make_ir(op = "ARG", data_type = "float*", value = "tensor_out", dependencies = []))
                tensor_pointers[i] = kernel_ir[-1]
            else:
                kernel_ir.append(make_ir(op = "ARG", data_type = "float*", value = f"tensor_{num}", dependencies = []))
                tensor_pointers[i] = kernel_ir[-1]
            op_tensors[i] = i._op
        elif isinstance(i._op, MovementOp):
            tensor_pointers[i] = tensor_pointers[i._parents[0]]
        elif isinstance(i._op, ReduceOp):
            if num == len(tensor_list) - 1:
                kernel_ir.append(make_ir(op = "ARG", data_type = "float*", value = "tensor_out", dependencies = []))
                tensor_pointers[i] = kernel_ir[-1]
            else:
                kernel_ir.append(make_ir(op = "ARG", data_type = "float*", value = f"tensor_{num}", dependencies = []))
                tensor_pointers[i] = kernel_ir[-1]
            op_tensors[i] = i._op

    for child in op_tensors:
        if isinstance(child._op, BinaryOp):
            bin_op_to_ir(child = child, kernel = kernel_ir, tensors = tensor_pointers, consts = const_pointers)
        elif isinstance(child._op, ReduceOp):
            red_op_to_ir(child = child, kernel = kernel_ir, tensors = tensor_pointers, consts = const_pointers)
    
    return kernel_ir