from __future__ import annotations
from copy import deepcopy
from ops import BinaryOp, MemoryOp, MovementOp
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


def separate_kernels(end: Tensor) -> list[list[Tensor]]:
    ast: list[Tensor] = end._topological_sort()

    kernel_tensors: list[list[Tensor]] = []
    temp_tensors: list[Tensor] = []

    for tensor in ast:
        temp_tensors.append(tensor)
        if isinstance(tensor._op, BinaryOp):
            kernel_tensors.append(deepcopy(temp_tensors))
            temp_tensors: list[Tensor] = []

    return kernel_tensors


def indexing_ir(tensor: Tensor, kernel: list[IR], dimensions: list[IR], tensor_pointers: dict[Tensor, IR], const_pointers: list[int, IR]) -> None:
    store_add: IR | None = None

    for index, dimension in enumerate(dimensions):
            if tensor._memory.stride[index] not in const_pointers:
                temp: IR = IR(op = "CONST", data_type = "int", value = tensor._memory.stride[index], dependencies = [])
                const_pointers[tensor._memory.stride[index]] = temp
                kernel.append(temp)
            temp_op: IR = IR(op = "MUL", data_type = "int", value = "", dependencies = [dimension, const_pointers[tensor._memory.stride[index]]])
            kernel.append(temp_op)
            if index != 0:
                temp_op: IR = IR(op = "ADD", data_type = "int", value = "", dependencies = [store_add, temp_op])
                kernel.append(temp_op)
            store_add = temp_op
    temp_load: IR = IR(op = "LOAD", data_type = "float", value = tensor_pointers[tensor].value, dependencies = [kernel[-1]])
    kernel.append(temp_load)


def preliminary_ir(ast_slice: list[Tensor]) -> list[IR]:
    kernel: list[IR] = []
    tensor_pointers: dict[Tensor, IR] = {}
    arg_counter = 0

    # Maybe as things are created, I can label them as load/store. This could help create directives down the line.
    # Further optimizations can be added that remove stores and replace them with loads if shapes match up/same operand.
    # Could be a curated second AST that indicates loads/stores and points to them. There can be overlap -> fusion in the future.
    #### I think I should alter this so that we take in AST slices instead of just a node. That way we can trace movement ops when 
    #### fusion does occur.
    for tensor in ast_slice:
        if tensor._parents[0] in tensor_pointers:
            tensor_pointers[tensor] = tensor_pointers[tensor._parents[0]]
        elif isinstance(tensor._op, BinaryOp):
            for parent in tensor._parents:
                temp: IR = IR(op = "ARG", data_type = parent._memory._data_type, value = f"operand_{arg_counter}", dependencies = [])
                tensor_pointers[parent] = temp
                kernel.append(temp)
                arg_counter += 1
            temp: IR = IR(op = "ARG", data_type = tensor._memory._data_type, value = "result", dependencies = [])
            tensor_pointers[tensor] = temp
            kernel.append(temp)

    global_shape: list[int] = head._parents[0]._memory.view
    dimensions: list[IR] = []
    const_pointers: dict[int, IR] = {0: IR(op = "CONST", data_type = "int", value = 0, dependencies = [])}
    kernel.append(const_pointers[0])

    for dimension in global_shape:
        if dimension not in const_pointers:
            const_pointers[dimension] = IR(op = "CONST", data_type = "int", value = dimension, dependencies = [])
            kernel.append(const_pointers[dimension])
        temp: IR = IR(op = "N-D", data_type = "", value = None, dependencies = [const_pointers[0], const_pointers[dimension]])
        dimensions.append(temp)
        kernel.append(temp)

    for parent in tensor_pointers:
        indexing_ir(tensor = parent, kernel = kernel, dimensions = dimensions, tensor_pointers = tensor_pointers, const_pointers = const_pointers)

    return kernel