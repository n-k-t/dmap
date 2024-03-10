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

class Lexer:
    def __init__(self, head: Tensor) -> None:
        self.tokens: list[Tensor] = self.separate_tensors(head)

    def separate_tensors(self, head: Tensor) -> list[Tensor]:
        dag: list[Tensor] = head._topological_sort()
        tensors: list[Tensor] = []
        for tensor in dag:
            if isinstance(tensor._op, BinaryOp) or isinstance(tensor._op, ReduceOp):
                tensors.append(tensor)
        return tensors

class Parser:
    def __init__(self, head: Tensor) -> None:
        self.token_stream: list[Tensor] = Lexer(head).tokens
        self.ast: list[list[IR]] = [self.preliminary_ir(token) for token in self.token_stream]

    # Replace the pointers here with a symbol table.
    # symbol_table: dict[str | Tensor, IR]
    def preliminary_ir(self, token: Tensor) -> list[IR]:
        ast: list[IR] = []
        tensor_pointers: dict[Tensor, IR] = {}
        ctx: dict[str, list[Tensor]] = {"LOAD": [], "STORE": []}

        for num, parent in enumerate(token._parents):
            temp: IR = IR("ARG", parent._memory._data_type + "*", f"operand_{num}", [])
            tensor_pointers[parent] = temp
            ast.append(temp)
            ctx["LOAD"].append(parent)
        temp: IR = IR("ARG", token._memory._data_type + "*", "result", [])
        tensor_pointers[token] = temp
        ast.append(temp)
        ctx["STORE"].append(token)

        global_shape: list[int] = token._parents[0]._memory.view
        dimensions: list[IR] = []
        const_pointers: dict[str, IR] = {str(0): IR("CONST", "int/float", str(0), [])}
        ast.append(const_pointers[str(0)])

        store_stride: list[int] = deepcopy(token._memory.stride)
        reduce_dim: int | None = None

        if isinstance(token._op, ReduceOp):
            reduce_dim: int = token._op.axis
            store_stride.insert(reduce_dim, 0)

        for num, dimension in enumerate(global_shape):
            if isinstance(token._op, ReduceOp) and num == reduce_dim:
                continue
            elif str(dimension) not in const_pointers:
                const_pointers[str(dimension)] = IR("CONST", "int", dimension, [])
                ast.append(const_pointers[str(dimension)])
            temp: IR = IR("N-D", "", f"axis_{num}", [const_pointers[str(0)], const_pointers[str(dimension)]])
            dimensions.append(temp)
            ast.append(temp)
        
        if isinstance(token._op, ReduceOp):
            if str(global_shape[reduce_dim]) not in const_pointers:
                const_pointers[str(global_shape[reduce_dim])] = IR("CONST", "int", global_shape[reduce_dim], [])
                ast.append(const_pointers[str(global_shape[reduce_dim])])
            temp: IR = IR("N-R", "", f"axis_{reduce_dim}", [const_pointers[str(0)], const_pointers[str(global_shape[reduce_dim])]])
            dimensions.insert(reduce_dim, temp)
            ast.append(temp)

        load_tracker: list[IR] = []

        for parent in ctx["LOAD"]:
            self.indexing_ir(parent, ast, dimensions, tensor_pointers, const_pointers, parent._memory.stride)
            if (len(parent._memory._mask["a"]) == 0) and (len(parent._memory._mask["p"]) == 0):
                load_tracker.append(ast[-1])
            else:
                load_tracker.append(ast[-1].dependencies[1].dependencies[0])

        temp_op: IR = IR(token._op.op, token._memory._data_type, "", [tensor for tensor in load_tracker])
        ast.append(temp_op)
        self.indexing_store_ir(token, ast, dimensions, tensor_pointers, const_pointers, store_stride)
        ast.append(IR("STORE", token._memory._data_type, tensor_pointers[token].value, [ast[-1], temp_op]))

        return ast

    def indexing_ir(self, tensor: Tensor, ast: list[IR], dimensions: list[IR], tensor_pointers: dict[Tensor, IR], const_pointers: dict[str, IR], stride: list[int]) -> None:
        store_add: IR = IR("NONE", "", "", [])

        for index, dimension in enumerate(dimensions):
                if str(stride[index]) not in const_pointers:
                    temp: IR = IR("CONST", "int", stride[index], [])
                    const_pointers[str(tensor._memory.stride[index])] = temp
                    ast.append(temp)
                temp_op: IR = IR("MUL", "int", "", [dimension, const_pointers[str(stride[index])]])
                ast.append(temp_op)
                if index != 0:
                    temp_op = IR("ADD", "int", "", [store_add, temp_op])
                    ast.append(temp_op)
                store_add = temp_op
        if tensor._memory._offset != 0:
            if str(tensor._memory._offset) not in const_pointers:
                temp = IR("CONST", "int", tensor._memory._offset, [])
                const_pointers[str(tensor._memory._offset)] = temp
                ast.append(temp)
            offset_op: IR = IR("ADD", "int", "", [temp_op, const_pointers[str(tensor._memory._offset)]])
            ast.append(offset_op)
        if (len(tensor._memory._mask["a"]) != 0) or (len(tensor._memory._mask["p"]) != 0):
            temp_load: IR = IR("LOAD", "float", tensor_pointers[tensor].value, [tensor_pointers[tensor], ast[-1]])
            ast.append(temp_load)

            comparison_count: int = 0
            for axis in tensor._memory._mask["a"]:
                if comparison_count > 0:
                    cmpr_holder: IR = ast[-1]
                if str(axis[2]) not in const_pointers:
                    temp = IR("CONST", "int", str(axis[2]), [])
                    const_pointers[str(axis[2])] = temp
                    ast.append(temp)
                temp_cmpr: IR = IR("CMPR", "", axis[1], [dimensions[axis[0]], const_pointers[str(axis[2])]])
                ast.append(temp_cmpr)
                if comparison_count > 0:
                    temp_or: IR = IR("OR", "", "", [cmpr_holder, temp_cmpr])
                    ast.append(temp_or)
                comparison_count += 1

            store_last: IR = ast[-1]
            temp_phi: IR = IR("PHI", "float", "phi_0", [])
            ast.append(temp_phi)
            store_1: IR = IR("STORE", "float", "phi_0", [temp_phi, temp_load])
            ast.append(store_1)
            store_2: IR = IR("STORE", "float", "phi_0", [temp_phi, const_pointers[str(0)]])
            ast.append(store_2)
            temp_redirect: IR = IR("IF/ELSE", "", "phi_0", [store_last, store_1, store_2])
            ast.append(temp_redirect)

        if (len(tensor._memory._mask["a"]) == 0) and (len(tensor._memory._mask["p"]) == 0):
            temp_load: IR = IR("LOAD", "float", tensor_pointers[tensor].value, [tensor_pointers[tensor], ast[-1]])
            ast.append(temp_load)


    def indexing_store_ir(self, tensor: Tensor, ast: list[IR], dimensions: list[IR], tensor_pointers: dict[Tensor, IR], const_pointers: dict[str, IR], stride: list[int]) -> None:
        store_add: IR | None = None

        for index, dimension in enumerate(dimensions):
                if str(stride[index]) not in const_pointers:
                    temp: IR = IR("CONST", "int", stride[index], [])
                    const_pointers[str(tensor._memory.stride[index])] = temp
                    ast.append(temp)
                temp_op: IR = IR("MUL", "int", "", [dimension, const_pointers[str(stride[index])]])
                ast.append(temp_op)
                if index != 0:
                    temp_op: IR = IR("ADD", "int", "", [store_add, temp_op])
                    ast.append(temp_op)
                store_add = temp_op

        temp_load: IR = IR("LOAD", "float", tensor_pointers[tensor].value, [tensor_pointers[tensor], ast[-1]])
        ast.append(temp_load)
