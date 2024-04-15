from ctypes import cast, c_float, CDLL, POINTER, c_size_t, c_void_p, sizeof
import subprocess
import tempfile
import time
import functools
import operator
from dmap.tensor import Op
from dmap.ir import IR, MuOp
from dmap.ops import Fusion

class C_IR:
    def __init__(self, ir, name: str, token: Op) -> None:
        self.ast = self._transpile(ir)
        self.name = name
        self._token = token
        self._const_bin_map = {MuOp.ADD: lambda a, b: a + b, 
                        MuOp.SUB: lambda a, b: a - b, 
                        MuOp.MUL: lambda a, b: a * b, 
                        MuOp.DIV: lambda a, b: a / b}
        self._bin_map = {MuOp.ADD: "+", MuOp.SUB: "-", MuOp.MUL: "*", MuOp.DIV: "/"}
        self._bin_set = {MuOp.ADD, MuOp.SUB, MuOp.MUL, MuOp.DIV}


    def _transpile(self, ast: list[IR]) -> list[IR]:
        for ir in ast:
            if ir.op == MuOp.N_D or ir.op == MuOp.N_R:
                ir.op = MuOp.LOOP

        return ast


    def to_code(self, branch: list[IR]) -> None:
        headers = ["#include <stdlib.h>\n"]
        macros = []
        main = [f"void {self.name}("]
        indent_level = 0
        store = ""

        red_set = {MuOp.SUM, MuOp.MAX, MuOp.MIN}

        un_set = {MuOp.EXP, MuOp.LOG}

        for ir in branch:
            if ir.op is MuOp.ARG:
                if "in" in ir.value:
                    if store == "":
                        store += f"{ir.dtype} {ir.value}"
                    else:
                        store += f", {ir.dtype} {ir.value}"
                else:
                    store += f", {ir.dtype} {ir.value})" + " {\n"
                    main.append(store)
                    store = ""
                    indent_level += 1
            elif ir.op is MuOp.TEMP:
                main.append(("\t" * indent_level) + f"{ir.dtype} {ir.value} = {ir.deps[0].value};\n")
            elif ir.op is MuOp.STORE:
                if ir.deps[1].op is MuOp.SUM:
                    eq_link = "+="
                else:
                    eq_link = "="
                if ir.deps[1].op in red_set:
                    if "#include <math.h>\n" not in headers:
                        headers.append("#include <math.h>\n")
                    if ir.deps[1].op is MuOp.MAX:
                        if "#define MAX(x, y) (((x) > (y)) ? (x) : (y))\n" not in macros:
                            macros.append("#define MAX(x, y) (((x) > (y)) ? (x) : (y))\n")
                    elif ir.deps[1].op is MuOp.MIN:
                        if "#define MIN(x, y) (((x) < (y)) ? (x) : (y))\n" not in macros:
                            macros.append("#define MIN(x, y) (((x) < (y)) ? (x) : (y))\n")
                    
                    main.append(("\t" * indent_level) + f"{ir.value} {eq_link} {self._render_reduce(ir.deps[1])};\n")
                elif ir.deps[1].op in self._bin_set:
                    main.append(("\t" * indent_level) + f"{self._render_load(ir.deps[0])} {eq_link} {self._render_binary(ir.deps[1])};\n")
                elif ir.deps[1].op in un_set:
                    if "#include <math.h>\n" not in headers:
                        headers.append("#include <math.h>\n")
                    main.append(("\t" * indent_level) + f"{self._render_load(ir.deps[0])} {eq_link} {self._render_unary(ir.deps[1])};\n")
                elif ir.deps[1].op is MuOp.TEMP:
                    main.append(("\t" * indent_level) + f"{self._render_load(ir.deps[0])} {eq_link} {ir.deps[1].value};\n")
            elif ir.op is MuOp.LOOP:
                main.append(("\t" * indent_level) + f"for (int {ir.value} = {ir.deps[0].value}; {ir.value} < {ir.deps[1].value}; {ir.value}++)" + " {\n")
                indent_level += 1
            elif ir.op is MuOp.END:
                indent_level -= 1
                main.append(("\t" * indent_level) + "}\n")
        
        main.append("}")

        return "".join(headers + macros + main)


    def _render_reduce(self, ir: IR) -> str:
        if ir.op is MuOp.SUM:
            return self._render_load(ir.deps[0])
        elif ir.op is MuOp.MAX:
            return f"MAX({ir.deps[0].value}, {self._render_load(ir.deps[1])})"
        elif ir.op is MuOp.MIN:
            return f"MIN({ir.deps[0].value}, {self._render_load(ir.deps[1])})"


    def _render_binary(self, ir: IR) -> str:
        if (ir.deps[0].op is MuOp.CONST) and (ir.deps[1].op is MuOp.CONST):
            new_val = self._const_bin_map[ir.op](int(ir.deps[0].value), int(ir.deps[1].value))
            if new_val == 0:
                return ""
            else:
                return f"({new_val})"
        elif ir.deps[0].op is MuOp.CONST:
            if ir.deps[0].value != 0:
                return f"{ir.deps[0].value} {self._bin_map[ir.op]} {self._render_load(ir.deps[1])}"
            elif ir.op is MuOp.SUB:
                return f"-{self._render_load(ir.deps[1])}"
            elif ir.op is MuOp.ADD:
                return f"{self._render_load(ir.deps[1])}"
            else:
                return "0"
        elif ir.deps[1].op is MuOp.CONST:
            if ir.deps[1].value != 0:
                return f"{self._render_load(ir.deps[0])} {self._bin_map[ir.op]} {ir.deps[1].value}"
            elif (ir.op is MuOp.SUB) or (ir.op is MuOp.ADD):
                return f"{self._render_load(ir.deps[0])}"
            elif ir.op is MuOp.MUL:
                return f"0"
            else:
                raise ValueError("You can't divide by 0.")
        else:
            return f"{self._render_load(ir.deps[0])} {self._bin_map[ir.op]} {self._render_load(ir.deps[1])}"


    def _render_unary(self, ir: IR) -> str:
        if ir.op is MuOp.EXP:
            return f"expf({self._render_load(ir.deps[0])})"
        elif ir.op is MuOp.LOG:
            return f"logf({self._render_load(ir.deps[0])})"


    @functools.lru_cache(maxsize = None, typed = False)
    def _render_load(self, ir: IR) -> str:
        if len(ir.deps) == 1:
            return f"(*({ir.deps[0].value}))"
        elif ir.deps[1].op is MuOp.CONST:
            if ir.deps[1].value == "0":
                return f"(*({ir.deps[0].value}))"
            else:
                return f"(*({ir.deps[0].value} + ({ir.deps[1].value})))"
        else:
            temp_ind = self._render_inds(ir.deps[1])
            if len(temp_ind) == 0:
                return f"(*({ir.deps[0].value}))"
            else:
                return f"(*({ir.deps[0].value} + {temp_ind}))"


    @functools.lru_cache(maxsize = None, typed = False)
    def _render_inds(self, ir: IR) -> str:
        if (ir.deps[0].op is MuOp.CONST) and (ir.deps[1].op is MuOp.CONST):
            new_val = self._const_bin_map[ir.op](int(ir.deps[0].value), int(ir.deps[1].value))
            if new_val == 0:
                return ""
            else:
                return f"({new_val})"

        if ir.deps[0].op in self._bin_set:
            l_expr = self._render_inds(ir.deps[0])
        else:
            l_expr = ir.deps[0].value
        if ir.deps[1].op in self._bin_set:
            r_expr = self._render_inds(ir.deps[1])
        else:
            r_expr = ir.deps[1].value

        expr = f"({l_expr} {self._bin_map[ir.op]} {r_expr})"
        return expr


    def bench(self, iters = 10, unset = False) -> None:
        name = self.name
        code = self.to_code(self.ast)
        token = self._token
        flop = token.num_flop
        program = self._compile(code, name)

        shapes = []

        if not isinstance(token.op, Fusion):
            for t in token.t_in:
                shapes.append(t.view)

            shapes.append(token.t_out.view)
        else:
            seen = []

            for fus in token.fus_ops:
                for t in fus.t_in:
                    if t not in seen:
                        shapes.append(t.view)
                        seen.append(t)

                if fus.t_out not in seen:
                    shapes.append(t.view)
                    seen.append(t)

        tensor_holder = []
        args_holder = []

        for shape in shapes:
            if unset:
                tensor_holder.append(self._gen_unset_vals(shape))
            else:
                tensor_holder.append(self._gen_random_vals(shape))
            args_holder.append(POINTER(c_float))

        program.argtypes = args_holder
        program.restype = None

        total_time = 0.0

        for _ in range(iters):
            start = time.perf_counter()

            program(*tensor_holder)

            stop = time.perf_counter()

            total_time += (stop - start)

        if unset:
            for tensor in tensor_holder:
                self._deallocate(tensor)

        avg_time = total_time / iters

        flops = flop / avg_time

        print(f"{name:<40} {avg_time:>10.3E} s {flops:>10.3E} FLOPs")


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


    def _gen_random_vals(self, size: list[int]) -> POINTER(c_float):
        size = functools.reduce(operator.mul, size)
        values = [hash(time.time_ns()) % 21 / 8 for _ in range(size)]
        return cast((c_float * size)(*values), POINTER(c_float))


    def _gen_unset_vals(self, size: list[int]) -> POINTER(c_float):
        func = CDLL("libc.so.6")["malloc"]
        func.argtypes = [c_size_t]
        func.restype = c_void_p
        size = functools.reduce(operator.mul, size)
        return cast(func(sizeof(c_float) * size), POINTER(c_float))


    def _deallocate(self, allocation) -> None:
        func = CDLL("libc.so.6")["free"]
        func.argtypes = [c_void_p]
        func.restype = None
        func(allocation)