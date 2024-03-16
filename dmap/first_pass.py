from __future__ import annotations
from copy import deepcopy
from dmap.ops import BinaryOp, ReduceOp
from dmap.tensor import Tensor
import functools
import operator

class IR:
    def __init__(self, op: str, data_type: str, value: str, dependencies: list[IR]) -> None:
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
        self.flop_count: list[int] = [self.calc_flop(tensor) for tensor in self.token_stream]
        self.ast: list[list[IR]] = [self.emit_ir(token) for token in self.token_stream]

    def calc_flop(self, tensor: Tensor) -> int:
        if isinstance(tensor._op, ReduceOp) and (tensor._parents[0]._memory.view[tensor._op.axis] > 1):
            op_adjustment: list[int] = deepcopy(tensor._parents[0]._memory.view)
            op_adjustment[tensor._op.axis] -= 1
            return functools.reduce(operator.mul, op_adjustment)
        else:
            return functools.reduce(operator.mul, tensor._parents[0]._memory.view)

    def discover_ctx(self, child: Tensor, ctx: dict[str, list[Tensor]], symbol_table: dict[str | Tensor, IR], ast: list[IR]) -> dict[str, list[Tensor]]:
        for num, parent in enumerate(child._parents):
            symbol_table[parent] = IR("ARG", parent._memory._data_type + "*", f"operand_{num}", [])
            ast.append(symbol_table[parent])
            ctx["LOAD"].append(parent)
        symbol_table[child] = IR("ARG", child._memory._data_type + "*", "result", [])
        ast.append(symbol_table[child])
        ctx["STORE"].append(child)

    # map the ND dimensions and add in the NR reduce dimension last if there is one.
    def map_dims(self, tensor: Tensor, symbol_table: dict[str | Tensor, IR], ast: list[IR]) -> None:
        pre_op_shape: list[int] = tensor._parents[0]._memory.view
        symbol_table["out_stride"] = deepcopy(tensor._memory.stride)
        symbol_table["loop_axes"] = []
        symbol_table[str("0")] = IR("CONST", "int", str(0), [])
        ast.append(symbol_table[str(0)])
        reduce_dim: int = -1
        if isinstance(tensor._op, ReduceOp):
            reduce_dim: int = tensor._op.axis
            symbol_table["out_stride"].insert(reduce_dim, 0)
        for num, dimension in enumerate(pre_op_shape):
            if isinstance(tensor._op, ReduceOp) and num == reduce_dim:
                continue
            elif str(dimension) not in symbol_table:
                symbol_table[str(dimension)] = IR("CONST", "int", dimension, [])
                ast.append(symbol_table[str(dimension)])
            non_reduce_axis: IR = IR("N-D", "", f"axis_{num}", [symbol_table[str(0)], symbol_table[str(dimension)]])
            symbol_table["loop_axes"].append(non_reduce_axis)
            ast.append(non_reduce_axis)
        if isinstance(tensor._op, ReduceOp):
            if str(pre_op_shape[reduce_dim]) not in symbol_table:
                symbol_table[str(pre_op_shape[reduce_dim])] = IR("CONST", "int", pre_op_shape[reduce_dim], [])
                ast.append(symbol_table[str(pre_op_shape[reduce_dim])])
            reduce_axis: IR = IR("N-R", "", f"axis_{reduce_dim}", [symbol_table[str(0)], symbol_table[str(pre_op_shape[reduce_dim])]])
            pre_op_shape.insert(reduce_dim, reduce_axis)
            ast.append(reduce_axis)
        pass

    def emit_ir(self, token: Tensor) -> list[IR]:
        ast: list[IR] = []
        symbol_table: dict[str | Tensor, IR | list[IR] | list[int]] = {}
        ctx: dict[str, list[Tensor]] = {"LOAD": [], "STORE": []}

        self.discover_ctx(token, ctx, symbol_table, ast)

        # Maybe make a dimension checker in case there is only one axis of size 1, so a scalar value?
        # self.map_dims(token, symbol_table, ast)

        # Place the dimensions within the symbol table as an ordered list of IR, labelled "axes" or something of the sort
        global_shape: list[int] = token._parents[0]._memory.view
        dimensions: list[IR] = []
        symbol_table[str("0")] = IR("CONST", "int", str(0), [])
        ast.append(symbol_table[str(0)])

        # Can we rename this as "out_stride" and then store in the symbol table?
        # Or remove entirely and don't insert a "0" in the non-existent dimension, can instead do it dynamically
        # at some point below as the tensor is passed to the function. Just need to track which axis to apply it to.
        # This could become harder in the future if things are getting reshaped within fused operations.
        store_stride: list[int] = deepcopy(token._memory.stride)
        reduce_dimension: int | None = None

        if isinstance(token._op, ReduceOp):
            reduce_dimension: int = token._op.axis
            store_stride.insert(reduce_dimension, 0)

        for num, dimension in enumerate(global_shape):
            if isinstance(token._op, ReduceOp) and num == reduce_dimension:
                continue
            elif str(dimension) not in symbol_table:
                symbol_table[str(dimension)] = IR("CONST", "int", dimension, [])
                ast.append(symbol_table[str(dimension)])
            non_reduce_axis: IR = IR("N-D", "", f"axis_{num}", [symbol_table[str(0)], symbol_table[str(dimension)]])
            dimensions.append(non_reduce_axis)
            ast.append(non_reduce_axis)
        
        if isinstance(token._op, ReduceOp):
            if str(global_shape[reduce_dimension]) not in symbol_table:
                symbol_table[str(global_shape[reduce_dimension])] = IR("CONST", "int", global_shape[reduce_dimension], [])
                ast.append(symbol_table[str(global_shape[reduce_dimension])])
            reduce_axis: IR = IR("N-R", "", f"axis_{reduce_dimension}", [symbol_table[str(0)], symbol_table[str(global_shape[reduce_dimension])]])
            dimensions.insert(reduce_dimension, reduce_axis)
            ast.append(reduce_axis)

        # Should be able to get rid of this by making updated "LOAD" entries in the symbol table    
        load_tracker: list[IR] = []

        for parent in ctx["LOAD"]:
            self.indexing_ir(parent, ast, dimensions, symbol_table, parent._memory.stride)
            if (len(parent._memory._mask["a"]) == 0) and (len(parent._memory._mask["p"]) == 0):
                load_tracker.append(ast[-1])
            else:
                load_tracker.append(ast[-1].dependencies[1].dependencies[0])

        temp_op: IR = IR(token._op.op, token._memory._data_type, "", [tensor for tensor in load_tracker])
        ast.append(temp_op)
        self.indexing_store_ir(token, ast, dimensions, symbol_table, store_stride)
        ast.append(IR("STORE", token._memory._data_type, symbol_table[token].value, [ast[-1], temp_op]))

        return ast
    
    def indexing_ir(self, tensor: Tensor, ast: list[IR], dimensions: list[IR], symbol_table: dict[str | Tensor, IR], stride: list[int]) -> None:
        store_add: IR = IR("NONE", "", "", [])

        for index, dimension in enumerate(dimensions):
                if str(stride[index]) not in symbol_table:
                    temp: IR = IR("CONST", "int", stride[index], [])
                    symbol_table[str(tensor._memory.stride[index])] = IR("CONST", "int", stride[index], [])
                    ast.append(temp)
                temp_op: IR = IR("MUL", "int", "", [dimension, symbol_table[str(stride[index])]])
                ast.append(temp_op)
                if index != 0:
                    temp_op = IR("ADD", "int", "", [store_add, temp_op])
                    ast.append(temp_op)
                store_add = temp_op
        if tensor._memory._offset != 0:
            if str(tensor._memory._offset) not in symbol_table:
                temp = IR("CONST", "int", tensor._memory._offset, [])
                symbol_table[str(tensor._memory._offset)] = temp
                ast.append(temp)
            offset_op: IR = IR("ADD", "int", "", [temp_op, symbol_table[str(tensor._memory._offset)]])
            ast.append(offset_op)
        if (len(tensor._memory._mask["a"]) != 0) or (len(tensor._memory._mask["p"]) != 0):
            temp_load: IR = IR("LOAD", "float", symbol_table[tensor].value, [symbol_table[tensor], ast[-1]])
            ast.append(temp_load)

            comparison_count: int = 0
            for axis in tensor._memory._mask["a"]:
                if comparison_count > 0:
                    cmpr_holder: IR = ast[-1]
                if str(axis[2]) not in symbol_table:
                    temp = IR("CONST", "int", str(axis[2]), [])
                    symbol_table[str(axis[2])] = temp
                    ast.append(temp)
                temp_cmpr: IR = IR("CMPR", "", axis[1], [dimensions[axis[0]], symbol_table[str(axis[2])]])
                ast.append(temp_cmpr)
                if comparison_count > 0:
                    temp_or: IR = IR("OR", "", "", [cmpr_holder, temp_cmpr])
                    ast.append(temp_or)
                comparison_count += 1

            phi_count: int = 0
            for ir in ast:
                if ir.op == "PHI":
                    phi_count += 1

            store_last: IR = ast[-1]
            temp_phi: IR = IR("PHI", "float", f"phi_{phi_count}", [])
            ast.append(temp_phi)
            store_1: IR = IR("STORE", "float", f"phi_{phi_count}", [temp_phi, temp_load])
            ast.append(store_1)
            store_2: IR = IR("STORE", "float", f"phi_{phi_count}", [temp_phi, symbol_table[str(0)]])
            ast.append(store_2)
            temp_redirect: IR = IR("IF/ELSE", "", f"phi_{phi_count}", [store_last, store_1, store_2])
            ast.append(temp_redirect)

        if (len(tensor._memory._mask["a"]) == 0) and (len(tensor._memory._mask["p"]) == 0):
            temp_load: IR = IR("LOAD", "float", symbol_table[tensor].value, [symbol_table[tensor], ast[-1]])
            ast.append(temp_load)


    def indexing_store_ir(self, tensor: Tensor, ast: list[IR], dimensions: list[IR], symbol_table: dict[str | Tensor, IR], stride: list[int]) -> None:
        store_add: IR | None = None

        for index, dimension in enumerate(dimensions):
                if str(stride[index]) not in symbol_table:
                    temp: IR = IR("CONST", "int", stride[index], [])
                    symbol_table[str(tensor._memory.stride[index])] = temp
                    ast.append(temp)
                temp_op: IR = IR("MUL", "int", "", [dimension, symbol_table[str(stride[index])]])
                ast.append(temp_op)
                if index != 0:
                    temp_op: IR = IR("ADD", "int", "", [store_add, temp_op])
                    ast.append(temp_op)
                store_add = temp_op

        temp_load: IR = IR("LOAD", "float", symbol_table[tensor].value, [symbol_table[tensor], ast[-1]])
        ast.append(temp_load)