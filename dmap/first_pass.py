from __future__ import annotations
from copy import deepcopy
from dmap.ops import BinaryOp, ReduceOp
from dmap.tensor import Tensor

class IR:
    def __init__(
            self,
            op: str,
            data_type: str,
            value: str,
            dependencies: list[IR]
        ) -> None:
        self.op: str = op
        self.data_type: str = data_type
        self.value: str = value
        self.dependencies: list[IR] = dependencies

    def __repr__(self) -> str:
        return f"OP: {self.op:>10},    DT: {self.data_type:>10},    VAL: {self.value:>10},    DEP: {[(i.op, i.value) for i in self.dependencies]}"

# Maybe remove this and isntead separate kernels is in a parser and everything else is in the lexer.
class Kernel:
    def __init__(self) -> None:
        self.ir: list[IR] = []

class Parser:
    def __init__(self) -> None:
        pass


class Lexer:
    def __init__(self) -> None:
        pass


def separate_kernels(end: Tensor) -> list[Tensor]:
    ast: list[Tensor] = end._topological_sort()

    kernel_tensors: list[Tensor] = []

    for tensor in ast:
        if isinstance(tensor._op, BinaryOp) or isinstance(tensor._op, ReduceOp):
            kernel_tensors.append(tensor)

    return kernel_tensors


def indexing_ir(tensor: Tensor, kernel: Kernel, dimensions: list[IR], tensor_pointers: dict[Tensor, IR], const_pointers: dict[str, IR], stride: list[int]) -> None:
    store_add: IR = IR(op = "NONE", data_type = "", value = "", dependencies = [])

    for index, dimension in enumerate(dimensions):
            if str(stride[index]) not in const_pointers:
                temp: IR = IR(op = "CONST", data_type = "int", value = stride[index], dependencies = [])
                const_pointers[str(tensor._memory.stride[index])] = temp
                kernel.ir.append(temp)
            temp_op: IR = IR(op = "MUL", data_type = "int", value = "", dependencies = [dimension, const_pointers[str(stride[index])]])
            kernel.ir.append(temp_op)
            if index != 0:
                temp_op = IR(op = "ADD", data_type = "int", value = "", dependencies = [store_add, temp_op])
                kernel.ir.append(temp_op)
            store_add = temp_op
    if tensor._memory._offset != 0:
        if str(tensor._memory._offset) not in const_pointers:
            temp = IR(op = "CONST", data_type = "int", value = tensor._memory._offset, dependencies = [])
            const_pointers[str(tensor._memory._offset)] = temp
            kernel.ir.append(temp)
        offset_op: IR = IR(op = "ADD", data_type = "int", value = "", dependencies = [temp_op, const_pointers[str(tensor._memory._offset)]])
        kernel.ir.append(offset_op)
    if (len(tensor._memory._mask["a"]) != 0) or (len(tensor._memory._mask["p"]) != 0):
        temp_load: IR = IR(op = "LOAD", data_type = "float", value = tensor_pointers[tensor].value, dependencies = [tensor_pointers[tensor], kernel.ir[-1]])
        kernel.ir.append(temp_load)

        comparison_count: int = 0
        for axis in tensor._memory._mask["a"]:
            if comparison_count > 0:
                cmpr_holder: IR = kernel.ir[-1]
            if str(axis[2]) not in const_pointers:
                temp = IR(op = "CONST", data_type = "int", value = axis[2], dependencies = [])
                const_pointers[str(axis[2])] = temp
                kernel.ir.append(temp)
            temp_cmpr: IR = IR(op = "CMPR", data_type = "", value = axis[1], dependencies = [dimensions[axis[0]], const_pointers[str(axis[2])]])
            kernel.ir.append(temp_cmpr)
            if comparison_count > 0:
                temp_or: IR = IR(op = "OR", data_type = "", value = "", dependencies = [cmpr_holder, temp_cmpr])
                kernel.ir.append(temp_or)
            comparison_count += 1

        store_last: IR = kernel.ir[-1]
        temp_phi: IR = IR(op = "PHI", data_type = "float", value = "phi_0", dependencies = [])
        kernel.ir.append(temp_phi)
        store_1: IR = IR(op = "STORE", data_type = "float", value = "phi_0", dependencies = [temp_phi, temp_load])
        kernel.ir.append(store_1)
        store_2: IR = IR(op = "STORE", data_type = "float", value = "phi_0", dependencies = [temp_phi, const_pointers[str(0)]])
        kernel.ir.append(store_2)
        temp_redirect: IR = IR(op = "IF/ELSE", data_type = "", value = "phi_0", dependencies = [store_last, store_1, store_2])
        kernel.ir.append(temp_redirect)

    if (len(tensor._memory._mask["a"]) == 0) and (len(tensor._memory._mask["p"]) == 0):
        temp_load: IR = IR(op = "LOAD", data_type = "float", value = tensor_pointers[tensor].value, dependencies = [tensor_pointers[tensor], kernel.ir[-1]])
        kernel.ir.append(temp_load)


def indexing_store_ir(tensor: Tensor, kernel: Kernel, dimensions: list[IR], tensor_pointers: dict[Tensor, IR], const_pointers: dict[str, IR], stride: list[int]) -> None:
    store_add: IR | None = None

    for index, dimension in enumerate(dimensions):
            if str(stride[index]) not in const_pointers:
                temp: IR = IR(op = "CONST", data_type = "int", value = stride[index], dependencies = [])
                const_pointers[str(tensor._memory.stride[index])] = temp
                kernel.ir.append(temp)
            temp_op: IR = IR(op = "MUL", data_type = "int", value = "", dependencies = [dimension, const_pointers[str(stride[index])]])
            kernel.ir.append(temp_op)
            if index != 0:
                temp_op: IR = IR(op = "ADD", data_type = "int", value = "", dependencies = [store_add, temp_op])
                kernel.ir.append(temp_op)
            store_add = temp_op

    temp_load: IR = IR(op = "LOAD", data_type = "float", value = tensor_pointers[tensor].value, dependencies = [tensor_pointers[tensor], kernel.ir[-1]])
    kernel.ir.append(temp_load)



def preliminary_ir(ast_slice: Tensor) -> list[IR]:
    kernel: Kernel = Kernel()
    tensor_pointers: dict[Tensor, IR] = {}
    control_flow: dict[str, list[Tensor]] = {"LOAD": [], "STORE": []}

    for num, parent in enumerate(ast_slice._parents):
        temp: IR = IR(op = "ARG", data_type = parent._memory._data_type + "*", value = f"operand_{num}", dependencies = [])
        tensor_pointers[parent] = temp
        kernel.ir.append(temp)
        control_flow["LOAD"].append(parent)
    temp: IR = IR(op = "ARG", data_type = ast_slice._memory._data_type + "*", value = "result", dependencies = [])
    tensor_pointers[ast_slice] = temp
    kernel.ir.append(temp)
    control_flow["STORE"].append(ast_slice)

    global_shape: list[int] = ast_slice._parents[0]._memory.view
    dimensions: list[IR] = []
    const_pointers: dict[str, IR] = {str(0): IR(op = "CONST", data_type = "int/float", value = 0, dependencies = [])}
    kernel.ir.append(const_pointers[str(0)])

    store_stride: list[int] = deepcopy(ast_slice._memory.stride)
    reduce_dim: int | None = None

    if isinstance(ast_slice._op, ReduceOp):
        reduce_dim: int = ast_slice._op.axis
        store_stride.insert(reduce_dim, 0)

    for num, dimension in enumerate(global_shape):
        if isinstance(ast_slice._op, ReduceOp) and num == reduce_dim:
            continue
        elif str(dimension) not in const_pointers:
            const_pointers[str(dimension)] = IR(op = "CONST", data_type = "int", value = dimension, dependencies = [])
            kernel.ir.append(const_pointers[str(dimension)])
        temp: IR = IR(op = "N-D", data_type = "", value = f"axis_{num}", dependencies = [const_pointers[str(0)], const_pointers[str(dimension)]])
        dimensions.append(temp)
        kernel.ir.append(temp)
    
    if isinstance(ast_slice._op, ReduceOp):
        if str(global_shape[reduce_dim]) not in const_pointers:
            const_pointers[str(global_shape[reduce_dim])] = IR(op = "CONST", data_type = "int", value = global_shape[reduce_dim], dependencies = [])
            kernel.ir.append(const_pointers[str(global_shape[reduce_dim])])
        temp: IR = IR(op = "N-R", data_type = "", value = f"axis_{reduce_dim}", dependencies = [const_pointers[str(0)], const_pointers[str(global_shape[reduce_dim])]])
        dimensions.insert(reduce_dim, temp)
        kernel.ir.append(temp)

    load_tracker: list[IR] = []

    for parent in control_flow["LOAD"]:
        indexing_ir(tensor = parent, kernel = kernel, dimensions = dimensions, tensor_pointers = tensor_pointers, const_pointers = const_pointers, stride = parent._memory.stride)
        if (len(parent._memory._mask["a"]) == 0) and (len(parent._memory._mask["p"]) == 0):
            load_tracker.append(kernel.ir[-1])
        else:
            load_tracker.append(kernel.ir[-1].dependencies[1].dependencies[0])

    temp_op: IR = IR(op = ast_slice._op.op, data_type = ast_slice._memory._data_type, value = "", dependencies = [tensor for tensor in load_tracker])
    kernel.ir.append(temp_op)
    indexing_store_ir(tensor = ast_slice, kernel = kernel, dimensions = dimensions, tensor_pointers = tensor_pointers, const_pointers = const_pointers, stride = store_stride)
    kernel.ir.append(IR(op = "STORE", data_type = ast_slice._memory._data_type, value = tensor_pointers[ast_slice].value, dependencies = [kernel.ir[-1], temp_op]))

    return kernel.ir