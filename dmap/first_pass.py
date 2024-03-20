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
        # Maybe create a parse tree (or syntax tree) the defines the kernel syntax and then this can be further passed along
        # to a semantic analyzer that essentially does what has already been outlined with the parser. The parse tree would 
        # likely be where fusion/mul-add operations come into play. This can all be combined into the parser, just have to think
        # through the structure.


    def calc_flop(self, tensor: Tensor) -> int:
        if isinstance(tensor._op, ReduceOp) and (tensor._parents[0]._memory.view[tensor._op.axis] > 1):
            op_adjustment: list[int] = deepcopy(tensor._parents[0]._memory.view)
            op_adjustment[tensor._op.axis] -= 1
            return functools.reduce(operator.mul, op_adjustment)
        else:
            return functools.reduce(operator.mul, tensor._parents[0]._memory.view)


    def discover_ctx(self, child: Tensor, ctx: dict[str, list[Tensor]], symbol_table: dict[str | Tensor, IR | list[IR] | list[int]], ast: list[IR]) -> dict[str, list[Tensor]]:
        for num, parent in enumerate(child._parents):
            symbol_table[parent] = IR("ARG", parent._memory._data_type + "*", f"operand_{num}", [])
            ast.append(symbol_table[parent])
            ctx["LOAD"].append(parent)
        symbol_table[child] = IR("ARG", child._memory._data_type + "*", "result", [])
        ast.append(symbol_table[child])
        ctx["STORE"].append(child)


    def map_dims(self, tensor: Tensor, symbol_table: dict[str | Tensor, IR | list[IR] | list[int]], ast: list[IR]) -> None:
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
            symbol_table["loop_axes"].insert(reduce_dim, reduce_axis)
            ast.append(reduce_axis)


    def from_view_and_shape(self, tensor: Tensor, ast: list[IR], symbol_table: dict[str | Tensor, IR | list[IR] | list[int]], ctx: dict[str, list[Tensor]]) -> None:
        if tensor in ctx["LOAD"]:
            stride = tensor._memory.stride
        else:
            stride = symbol_table["out_stride"]
        
        store_add: IR = IR("NONE", "", "", [])

        for index, dimension in enumerate(symbol_table["loop_axes"]):
            if str(stride[index]) not in symbol_table:
                symbol_table[str(stride[index])] = IR("CONST", "int", stride[index], [])
                ast.append(symbol_table[str(stride[index])])
            temp_op: IR = IR("MUL", "int", "", [dimension, symbol_table[str(stride[index])]])
            ast.append(temp_op)
            if index != 0:
                temp_op = IR("ADD", "int", "", [store_add, temp_op])
                ast.append(temp_op)
            store_add = temp_op


    def account_for_offset(self, tensor: Tensor, ast: list[IR], symbol_table: dict[str | Tensor, IR | list[IR] | list[int]]) -> None:
        look_back: int = -1
        if str(tensor._memory._offset) not in symbol_table:
                symbol_table[str(tensor._memory._offset)] = IR("CONST", "int", tensor._memory._offset, [])
                ast.append(symbol_table[str(tensor._memory._offset)])
                look_back -= 1
        ast.append(IR("ADD", "int", "", [ast[look_back], symbol_table[str(tensor._memory._offset)]]))


    def render_mask(self, tensor: Tensor, ast: list[IR], symbol_table: dict[str | Tensor, IR | list[IR] | list[int]]) -> None:
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
            temp_cmpr: IR = IR("CMPR", "", axis[1], [symbol_table["loop_axes"][axis[0]], symbol_table[str(axis[2])]])
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


    def find_indices(self, tensor: Tensor, ast: list[IR], symbol_table: dict[str | Tensor, IR | list[IR] | list[int]], ctx: dict[str, list[Tensor]]) -> None:
        self.from_view_and_shape(tensor, ast, symbol_table, ctx)

        if tensor._memory._offset != 0:
            self.account_for_offset(tensor, ast, symbol_table)
            
        if (len(tensor._memory._mask["a"]) != 0) or (len(tensor._memory._mask["p"]) != 0):
            self.render_mask(tensor, ast, symbol_table)

        if (len(tensor._memory._mask["a"]) == 0) and (len(tensor._memory._mask["p"]) == 0):
            temp_load: IR = IR("LOAD", "float", symbol_table[tensor].value, [symbol_table[tensor], ast[-1]])
            ast.append(temp_load)


    def emit_ir(self, token: Tensor) -> list[IR]:
        ast: list[IR] = []
        symbol_table: dict[str | Tensor, IR | list[IR] | list[int]] = {}
        ctx: dict[str, list[Tensor]] = {"LOAD": [], "STORE": []}

        self.discover_ctx(token, ctx, symbol_table, ast)

        self.map_dims(token, symbol_table, ast)

        symbol_table["LOADED"]: list[IR] = []
        for parent in ctx["LOAD"]:
            self.find_indices(parent, ast, symbol_table, ctx)
            if (len(parent._memory._mask["a"]) == 0) and (len(parent._memory._mask["p"]) == 0):
                symbol_table["LOADED"].append(ast[-1])
            else:
                symbol_table["LOADED"].append(ast[-1].dependencies[1].dependencies[0])

        temp_op: IR = IR(token._op.op, token._memory._data_type, "", [tensor for tensor in symbol_table["LOADED"]])
        ast.append(temp_op)
        self.find_indices(token, ast, symbol_table, ctx)
        ast.append(IR("STORE", token._memory._data_type, symbol_table[token].value, [ast[-1], temp_op]))

        return ast