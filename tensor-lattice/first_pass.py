from __future__ import annotations
from copy import deepcopy
from ops import BinaryOp, ReduceOp
from tensor import Tensor

class IR:
    def __init__(
            self,
            op: str,
            data_type: str,
            value: int | float | str,
            dependencies: list[IR]
        ) -> None:
        self.op: str = op
        self.data_type: str = data_type
        self.value: int | float | str = value
        self.dependencies: list[IR] = dependencies

    def __repr__(self) -> str:
        return f"OP: {self.op:>10},    DT: {self.data_type:>10},    VAL: {str(self.value):>10},    DEP: {[(i.op, i.value) for i in self.dependencies]}"


def separate_kernels(end: Tensor) -> list[Tensor]:
    ast: list[Tensor] = end._topological_sort()

    kernel_tensors: list[Tensor] = []

    for tensor in ast:
        if isinstance(tensor._op, BinaryOp) or isinstance(tensor._op, ReduceOp):
            kernel_tensors.append(tensor)

    return kernel_tensors


def indexing_ir(tensor: Tensor, kernel: list[IR], dimensions: list[IR], tensor_pointers: dict[Tensor, IR], const_pointers: list[int, IR], stride: list[int]) -> None:
    store_add: IR | None = None

    for index, dimension in enumerate(dimensions):
            if stride[index] not in const_pointers:
                temp: IR = IR(op = "CONST", data_type = "int", value = stride[index], dependencies = [])
                const_pointers[tensor._memory.stride[index]] = temp
                kernel.append(temp)
            temp_op: IR = IR(op = "MUL", data_type = "int", value = "", dependencies = [dimension, const_pointers[stride[index]]])
            kernel.append(temp_op)
            if index != 0:
                temp_op: IR = IR(op = "ADD", data_type = "int", value = "", dependencies = [store_add, temp_op])
                kernel.append(temp_op)
            store_add = temp_op
    temp_load: IR = IR(op = "LOAD", data_type = "float", value = tensor_pointers[tensor].value, dependencies = [tensor_pointers[tensor], kernel[-1]])
    kernel.append(temp_load)


def preliminary_ir(ast_slice: Tensor) -> list[IR]:
    kernel: list[IR] = []
    tensor_pointers: dict[Tensor, IR] = {}
    control_flow: dict[str, list[Tensor]] = {"LOAD": [], "STORE": []}

    for num, parent in enumerate(ast_slice._parents):
        temp: IR = IR(op = "ARG", data_type = parent._memory._data_type + "*", value = f"operand_{num}", dependencies = [])
        tensor_pointers[parent] = temp
        kernel.append(temp)
        control_flow["LOAD"].append(parent)
    temp: IR = IR(op = "ARG", data_type = ast_slice._memory._data_type + "*", value = "result", dependencies = [])
    tensor_pointers[ast_slice] = temp
    kernel.append(temp)
    control_flow["STORE"].append(ast_slice)

    global_shape: list[int] = ast_slice._parents[0]._memory.view
    dimensions: list[IR] = []
    const_pointers: dict[int, IR] = {0: IR(op = "CONST", data_type = "int", value = 0, dependencies = [])}
    kernel.append(const_pointers[0])

    store_stride: list[int] = deepcopy(ast_slice._memory.stride)
    reduce_dim: int | None = None

    if isinstance(ast_slice._op, ReduceOp):
        reduce_dim: int = ast_slice._op.axis
        store_stride.insert(reduce_dim, 0)

    for num, dimension in enumerate(global_shape):
        if isinstance(ast_slice._op, ReduceOp) and num == reduce_dim:
            continue
        elif dimension not in const_pointers:
            const_pointers[dimension] = IR(op = "CONST", data_type = "int", value = dimension, dependencies = [])
            kernel.append(const_pointers[dimension])
        temp: IR = IR(op = "N-D", data_type = "", value = f"axis_{num}", dependencies = [const_pointers[0], const_pointers[dimension]])
        dimensions.append(temp)
        kernel.append(temp)
    
    if isinstance(ast_slice._op, ReduceOp):
        if global_shape[reduce_dim] not in const_pointers:
            const_pointers[global_shape[reduce_dim]] = IR(op = "CONST", data_type = "int", value = global_shape[reduce_dim], dependencies = [])
            kernel.append(const_pointers[global_shape[reduce_dim]])
        temp: IR = IR(op = "N-R", data_type = "", value = f"axis_{reduce_dim}", dependencies = [const_pointers[0], const_pointers[global_shape[reduce_dim]]])
        dimensions.insert(reduce_dim, temp)
        kernel.append(temp)

    load_tracker: list[IR] = []

    for parent in control_flow["LOAD"]:
        indexing_ir(tensor = parent, kernel = kernel, dimensions = dimensions, tensor_pointers = tensor_pointers, const_pointers = const_pointers, stride = parent._memory.stride)
        load_tracker.append(kernel[-1])

    temp_op: IR = IR(op = ast_slice._op.op, data_type = ast_slice._memory._data_type, value = "", dependencies = [tensor for tensor in load_tracker])
    kernel.append(temp_op)
    indexing_ir(tensor = ast_slice, kernel = kernel, dimensions = dimensions, tensor_pointers = tensor_pointers, const_pointers = const_pointers, stride = store_stride)
    kernel.append(IR(op = "STORE", data_type = ast_slice._memory._data_type, value = tensor_pointers[ast_slice].value, dependencies = [kernel[-1], temp_op]))

    return kernel