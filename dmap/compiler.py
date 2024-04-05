from copy import deepcopy
from typing import Callable
from ctypes import cast, c_float, CDLL, POINTER
import subprocess
import tempfile
import time
import functools
import operator
from dmap.tensor import Tensor, Op
from dmap.ops import Reduce, Binary, Movement, Memory, Fusion, Unary
from dmap.ir import IR, MuOp


class Compiler():
    def __init__(self, head: Tensor, fuse: bool = False) -> None:
        self.tokens = self._traverse(head)
        if fuse:
            self.tokens = self._fuse(self.tokens)
        self.kernel_flop = [token.num_flop for token in self.tokens]
        self.kernel_names: list[str] = []
        self._kernel_tensors: list[list[list[int]]] = []
        self._op_pattern_match: dict[Binary|Reduce|Unary, Callable] = {
                            Binary.ADD: lambda ast, symbol_table: self._render_binary(MuOp.ADD, ast, symbol_table), 
                            Binary.SUB: lambda ast, symbol_table: self._render_binary(MuOp.SUB, ast, symbol_table), 
                            Binary.MUL: lambda ast, symbol_table: self._render_binary(MuOp.MUL, ast, symbol_table), 
                            Binary.DIV: lambda ast, symbol_table: self._render_binary(MuOp.DIV, ast, symbol_table), 
                            Reduce.SUM: lambda ast, symbol_table: self._render_reduce(MuOp.SUM, ast, symbol_table), 
                            Reduce.MAX: lambda ast, symbol_table: self._render_reduce(MuOp.MAX, ast, symbol_table), 
                            Reduce.MIN: lambda ast, symbol_table: self._render_reduce(MuOp.MIN, ast, symbol_table), 
                            Unary.EXP: lambda ast, symbol_table: self._render_unary(MuOp.EXP, ast, symbol_table), 
                            Unary.LOG: lambda ast, symbol_table: self._render_unary(MuOp.LOG, ast, symbol_table)
                        }
        self.abstract_ast = [self._lower(token) for token in self.tokens]
        self.specific_ast = [self._transpile(tree) for tree in self.abstract_ast]
        self.code = [self._codegen(tree, name) for tree, name in zip(self.specific_ast, self.kernel_names)]
        self.program = [self._compile(text, name) for text, name in zip(self.code, self.kernel_names)]



    def _apply_restrictions(self, token_stream: list[Op]) -> list[Op]:
        for token in token_stream:
            if isinstance(token.op, Memory) or isinstance(token.op, Movement):
                pass
            elif isinstance(token.op, Fusion):
                for tensor in token.t_in:
                    if token.t_out.dtype != tensor.dtype:
                        raise ValueError("Fused operations must have the same data type, or be typecast.")
        
        return token_stream


    def _traverse(self, tensor: Tensor) -> list[Op]:
        def top_sort(tensor, visited, stack) -> list[Tensor]:
            visited.add(tensor)
            for parent in tensor.parents:
                if parent not in visited:
                    top_sort(parent, visited, stack)
            if tensor.op not in visited:
                visited.add(tensor.op)
                stack.append(tensor.op)
            return stack
        return self._apply_restrictions(top_sort(tensor, set(), []))


    def _fuse(self, ops: list[Op]) -> list[Op]:
        index: int = 0

        while index < len(ops):
            if isinstance(ops[index].op, Binary) and isinstance(ops[index + 1].op, Reduce) and (ops[index].t_out in ops[index + 1].t_in):
                op_holder = Op(Fusion.ELE_RED)
                op_holder.num_flop = ops[index].num_flop + ops[index + 1].num_flop
                op_holder.fus_ops = [ops[index], ops[index + 1]]
                ops.pop(index)
                ops[index] = op_holder
            index += 1
        
        return ops


    def _lower(self, token: Op) -> list[IR]:
        if isinstance(token.op, Memory) or isinstance(token.op, Movement):
            self._kernel_tensors.append([[]])
            self.kernel_names.append("Free")
            return [IR(MuOp.NOOP, "", "", [])]

        temp_name: str = self._gen_kernel_name(token) + "_v"

        repetitions: int = 0
        for name in self.kernel_names:
            if temp_name in name:
                repetitions += 1
        
        self.kernel_names.append(temp_name + f"{repetitions}")

        queue: list[Op] = self._enqueue(token)
        
        ast: list[IR] = []
        ctx: dict[str, list[Tensor]] = {"LOAD": [], "TEMP": [], "STORE": []}
        symbol_table: dict[Tensor|str, IR|list[IR]] = {}

        self._define_ctx(queue, ast, ctx, symbol_table)

        self._get_global_dim(ast, ctx, symbol_table)

        for op in queue:
            if isinstance(op.op, Reduce):
                self._make_and_move_reduce_axis(ast, op.axis, symbol_table["global_dim"])

            symbol_table["in_tensor_pointers"] = []
            symbol_table["out_tensor_pointers"] = IR(MuOp.NOOP, "", "", [])

            for parent in op.t_in:
                symbol_table["in_tensor_pointers"].append(self._index_in(parent, ast, symbol_table))

            self._op_pattern_match[op.op](ast, symbol_table)

            if isinstance(op.op, Reduce):
                symbol_table["out_tensor_pointers"] = self._index_in(op.t_out, ast, symbol_table, op.axis)
            else:
                symbol_table["out_tensor_pointers"] = self._index_in(op.t_out, ast, symbol_table)

            ast.append(IR(MuOp.STORE, symbol_table["out_tensor_pointers"].dtype, symbol_table["out_tensor_pointers"].value, 
                        [symbol_table["out_tensor_pointers"], symbol_table["linking_pointer"]]))
            
        self._end_dims(ast, symbol_table)

        return ast


    def _gen_kernel_name(self, op: Op) -> str:
        if isinstance(op.op, Memory) or isinstance(op.op, Movement):
            return "NoOp"
        elif isinstance(op.op, Unary) or isinstance(op.op, Binary):
            return "_".join(["El"] + [str(dim) for dim in op.t_in[0].view] + ["to"] + [str(dim) for dim in op.t_out.view])
        elif isinstance(op.op, Reduce):
            return "_".join(["Re"] + [str(dim) for dim in op.t_in[0].view] + ["to"] + [str(dim) for dim in op.t_out.view])
        elif isinstance(op.op, Fusion):
            return "_".join(["Fu"] + [str(dim) for dim in op.fus_ops[0].t_in[0].view] + ["to"] + [str(dim) for dim in op.fus_ops[-1].t_out.view])


    def _enqueue(self, op: Op) -> list[Op]:
        if isinstance(op.op, Fusion):
            return op.fus_ops
        else:
            return [op]


    def _define_ctx(self, queue: list[Op], ast: list[IR], ctx: dict[str, list[Tensor]], symbol_table: dict[Tensor|str, IR|list[IR]]) -> None:
        tensor_count: int = 0
        tensor_sizes = []

        for op in queue:
            for parent in op.t_in:
                if parent in ctx["STORE"]:
                    symbol_table[parent].value = f"operand_{tensor_count}"
                    ctx["LOAD"].append(parent)
                else:
                    symbol_table[parent] = IR(MuOp.ARG, parent.dtype + "*", f"operand_{tensor_count}", [])
                    ast.append(symbol_table[parent])
                    ctx["LOAD"].append(parent)
                    tensor_sizes.append(parent.view)
                tensor_count += 1
            
            symbol_table[op.t_out] = IR(MuOp.ARG, op.t_out.dtype + "*", "result", [])
            ast.append(symbol_table[op.t_out])
            ctx["STORE"].append(op.t_out)
            tensor_sizes.append(op.t_out.view)
        
        self._kernel_tensors.append(tensor_sizes)


    def _get_global_dim(self, ast: list[IR], ctx: dict[str, list[Tensor]], symbol_table: dict[Tensor|str, IR|list[IR]]) -> None:
        symbol_table["global_dim"] = []

        if "0" not in symbol_table:
            symbol_table["0"] = IR(MuOp.CONST, "int", "0", [])
            ast.append(symbol_table["0"])

        for num, dim in enumerate(ctx["LOAD"][0].view):
            if str(dim) not in symbol_table:
                symbol_table[str(dim)] = IR(MuOp.CONST, "int", str(dim), [])
                ast.append(symbol_table[str(dim)])
            
            symbol_table["global_dim"].append(IR(MuOp.N_D, "", f"axis_{num}", [symbol_table["0"], symbol_table[str(dim)]]))
            ast.append(symbol_table["global_dim"][-1])

    
    def _make_and_move_reduce_axis(self, ast: list[IR], index: int, dimensions: list[IR]):
        dimensions[index].op = MuOp.N_R
        reduce_axis_index = ast.index(dimensions[index])
        inner_axis_index = max([ast.index(dim) for dim in dimensions])
        if reduce_axis_index != inner_axis_index:
            holder = ast[reduce_axis_index]
            ast[reduce_axis_index] = ast[inner_axis_index]
            ast[inner_axis_index] = holder


    def _index_in(self, tensor: Tensor, ast: list[IR], symbol_table: dict[Tensor|str, IR|list[IR]], red_axis: int = -1) -> IR:
        self._map_dims(tensor, ast, symbol_table, red_axis)

        return ast[-1]


    def _map_dims(self, tensor: Tensor, ast: list[IR], symbol_table: dict[Tensor|str, IR|list[IR]], red_axis: int) -> None:
        stride: list[int] = deepcopy(tensor.stride)
        if red_axis > -1:
            stride.insert(red_axis, 0)
        previous_term: IR = IR(MuOp.NOOP, "", "", [])

        update_prev_term_index: int = 0

        for index, dim in enumerate(symbol_table["global_dim"]):
            if index == red_axis:
                update_prev_term_index += 1
                continue
            elif str(stride[index]) not in symbol_table:
                symbol_table[str(stride[index])] = IR(MuOp.CONST, "int", stride[index], [])
                ast.append(symbol_table[str(stride[index])])
            temp_op: IR = IR(MuOp.MUL, "int", "", [dim, symbol_table[str(stride[index])]])
            ast.append(temp_op)
            if index != update_prev_term_index:
                temp_op = IR(MuOp.ADD, "int", "", [previous_term, temp_op])
                ast.append(temp_op)
            previous_term = temp_op

        ast.append(IR(MuOp.LOAD, tensor.dtype, symbol_table[tensor].value, [symbol_table[tensor], ast[-1]]))


    def _render_binary(self, op: MuOp, ast: list[IR], symbol_table: dict[Tensor|str, IR|list[IR]]) -> None:
        tensors: list[IR] = symbol_table["in_tensor_pointers"]

        symbol_table["linking_pointer"] = IR(op, tensors[0].dtype, "", tensors)

        ast.append(symbol_table["linking_pointer"])


    def _find_ir_parents(self, ir: IR, ast: list[IR]) -> set[IR]:
        parent_set: set[IR] = set([ir])

        delta_len: int = len(parent_set)

        while delta_len != 0:
            start_len: int = len(parent_set)
            for ir in ast:
                if parent_set.intersection([i for i in ir.deps]):
                    parent_set.add(ir)
            delta_len = len(parent_set) - start_len

        return parent_set


    def _find_ir_children(self, ir: IR) -> set[IR]:
        return set.union(set(ir.deps), *[self._find_ir_children(dep) for dep in ir.deps])


    def _render_reduce(self, op: MuOp, ast: list[IR], symbol_table: dict[Tensor|str, IR|list[IR]]) -> None:
        match_init_val: dict[MuOp, str] = {MuOp.SUM: "0", MuOp.MAX: "-INFINITY", MuOp.MIN: "INFINITY"}
        red_init_val: str = match_init_val[op]
        tensors: list[IR] = symbol_table["in_tensor_pointers"]

        deps: set[IR] = self._find_ir_children(tensors[0])

        temp_index: int = max([ast.index(dep) for dep in deps if dep.op == MuOp.N_R])

        if red_init_val not in symbol_table:
            symbol_table[red_init_val] = IR(MuOp.CONST, "int", red_init_val, [])
            ast.insert(temp_index, symbol_table[red_init_val])
            temp_index += 1

        temp_count: int = 0
        for ir in ast:
            if ir.op == MuOp.TEMP:
                temp_count += 1

        symbol_table["linking_pointer"] = IR(MuOp.TEMP, tensors[0].dtype, f"temp_{temp_count}", [symbol_table[red_init_val]])
        ast.insert(temp_index, symbol_table["linking_pointer"])

        if op == MuOp.SUM:
            ast.append(IR(op, tensors[0].dtype, "", tensors))
        else:
            ast.append(IR(op, tensors[0].dtype, "", [symbol_table["linking_pointer"]] + tensors))
        ast.append(IR(MuOp.STORE, tensors[0].dtype, symbol_table["linking_pointer"].value, [symbol_table["linking_pointer"], ast[-1]]))


    def _render_unary(self, op: MuOp, ast: list[IR], symbol_table: dict[Tensor|str, IR|list[IR]]) -> None:
        tensors: list[IR] = symbol_table["in_tensor_pointers"]

        symbol_table["linking_pointer"] = IR(op, tensors[0].dtype, "", tensors)

        ast.append(symbol_table["linking_pointer"])


    def _end_dims(self, ast: list[IR], symbol_table: dict[Tensor|str, IR|list[IR]]) -> None:
        for axis in symbol_table["global_dim"]:
            end_index = max([ast.index(ir) for ir in self._find_ir_parents(axis, ast)]) + 1
            ast.insert(end_index, IR(MuOp.END, "", axis.value, [axis]))


    # -------------Change the IR style------------- #
    def _transpile(self, ast: list[IR]) -> list[IR]:
        for ir in ast:
            if ir.op == MuOp.N_D or ir.op == MuOp.N_R:
                ir.op = MuOp.LOOP

        return ast


    # -------------Render the IR as code------------- #
    def _gen_tensor_indices(self, ir: IR) -> str:
        map_ops: dict[str, str] = {MuOp.ADD: "+", MuOp.MUL: "*"}
        internal_ops: list[str] = [MuOp.ADD, MuOp.MUL]
        if ir.deps[0].op in internal_ops:
            left_exp = self._gen_tensor_indices(ir.deps[0])
        else:
            left_exp = ir.deps[0].value
        if ir.deps[1].op in internal_ops:
            right_exp = self._gen_tensor_indices(ir.deps[1])
        else:
            right_exp = ir.deps[1].value
        
        expression = f"({left_exp} {map_ops[ir.op]} {right_exp})"

        return expression


    def _gen_load(self, ir: IR) -> str:
        return f"(*({ir.value} + " + self._gen_tensor_indices(ir.deps[1]) + "))"


    def _gen_store(self, ir: IR) -> str:
        map_ops: dict[str, str] = {MuOp.ADD: " + ", MuOp.MUL: " * ", MuOp.SUB: " - ", MuOp.DIV: " / "}

        if ir.deps[0].op == MuOp.TEMP:
            left = ir.deps[0].value
        elif ir.deps[0].op == MuOp.LOAD:
            left = self._gen_load(ir.deps[0])

        if ir.deps[1].op == MuOp.TEMP:
            left += " = "
            right = ir.deps[1].value + ";\n"
        elif ir.deps[1].op == MuOp.SUM:
            left += " += "
            right = self._gen_load(ir.deps[1].deps[0]) + ";\n"
        elif ir.deps[1].op == MuOp.MAX:
            left += " = "
            right = "MAX(" + ir.deps[1].deps[0].value + ", " + self._gen_load(ir.deps[1].deps[1]) + ");\n"
        elif ir.deps[1].op == MuOp.MIN:
            left += " = "
            right = "MIN(" + ir.deps[1].deps[0].value + ", " + self._gen_load(ir.deps[1].deps[1]) + ");\n"
        elif ir.deps[1].op == MuOp.EXP:
            left += " = "
            right = "expf(" + self._gen_load(ir.deps[1].deps[0]) + ");\n"
        elif ir.deps[1].op == MuOp.LOG:
            left += " = "
            right = "logf(" + self._gen_load(ir.deps[1].deps[0]) + ");\n"
        else:
            left += " = "
            right = self._gen_load(ir.deps[1].deps[0]) + map_ops[ir.deps[1].op] + self._gen_load(ir.deps[1].deps[1]) + ";\n"

        return (left + right)


    def _codegen(self, ast: list[IR], name: str) -> str:
        indent_level: int = 0
        headers: list[str] = ["#include <stdlib.h>\n"]
        macros: list[str] = []
        main: list[str] = [f"void {name}("]

        for ir in ast:
            if ir.op == MuOp.NOOP:
                return "N/A"
            elif ir.op == MuOp.ARG:
                main.append(f"{ir.dtype} {ir.value}")
                if ir.value == "result":
                    main.append(") {\n")
                    indent_level += 1
                else:
                    main.append(", ")
            elif ir.op == MuOp.LOOP:
                main.append(("\t" * indent_level) + f"for (int {ir.value} = {ir.deps[0].value}; {ir.value} < {ir.deps[1].value}; {ir.value}++)" + " {\n")
                indent_level += 1
            elif ir.op == MuOp.END:
                indent_level -= 1
                main.append(("\t" * indent_level) + "}\n")
            elif ir.op == MuOp.MAX:
                if "#include <math.h>\n" not in headers:
                    headers.append("#include <math.h>\n")
                macros.append("#define MAX(x, y) (((x) > (y)) ? (x) : (y))\n")
            elif ir.op == MuOp.MIN:
                if "#include <math.h>\n" not in headers:
                    headers.append("#include <math.h>\n")
                macros.append("#define MIN(x, y) (((x) < (y)) ? (x) : (y))\n")
            elif ir.op == MuOp.TEMP:
                main.append(("\t" * indent_level) + f"{ir.dtype} {ir.value} = {ir.deps[0].value};\n")
            elif (ir.op == MuOp.EXP) or (ir.op == MuOp.LOG):
                if "#include <math.h>\n" not in headers:
                    headers.append("#include <math.h>\n")
            elif ir.op == MuOp.STORE:
                main.append(("\t" * indent_level) + self._gen_store(ir))    

        if len(macros) == 0:
            return "".join(headers + ["\n"] + main + ["}"])
        else:
            return "".join(headers + ["\n"] + macros + ["\n"] + main + ["}"])


    # -------------Compile the code to a .so file------------- #
    def _compile(self, code: str, name: str) -> bytes:
        if name == "Free":
            return b''
        
        with tempfile.NamedTemporaryFile(delete = True) as code_file:
            with open(code_file.name, 'w') as file:
                file.write(code)

            with tempfile.NamedTemporaryFile(delete = True) as byte_file:
                process = subprocess.run([f"gcc -x c -shared -O3 -Wall -Werror -o {byte_file.name} -fPIC {code_file.name}"], shell = True, capture_output = True)
                if process.returncode != 0:
                    print(process.stderr.decode())
                    raise RuntimeError("There was an error while compiling a kernel.")
                
                return CDLL(byte_file.name)[name]            


    # -------------Benchmark the code------------- #
    def _gen_random_vals(self, size: list[int]) -> POINTER(c_float):
        size = functools.reduce(operator.mul, size)
        values = [hash(time.time_ns()) % 21 / 8 for _ in range(size)]
        return cast((c_float * size)(*values), POINTER(c_float))


    def bench(self) -> None:
        for name, shapes, prog, flop in zip(self.kernel_names, self._kernel_tensors, self.program, self.kernel_flop):
            if name == "Free":
                print(f"{name:<40} {0.0:>10.3f} s {0.0:>10.3E} FLOPs")
                continue

            tensor_holder = []
            args_holder = []

            for shape in shapes:
                tensor_holder.append(self._gen_random_vals(shape))
                args_holder.append(POINTER(c_float))

            prog.argtypes = args_holder
            prog.restype = None

            total_time = 0.0
            iters = 100

            for _ in range(iters):
                start = time.perf_counter()

                prog(*tensor_holder)

                stop = time.perf_counter()

                total_time += (stop - start)

            avg_time = total_time / iters

            flops = flop / avg_time

            print(f"{name:<40} {avg_time:>10.3E} s {flops:>10.3E} FLOPs")