from copy import deepcopy
from ctypes import cast, c_float, CDLL, POINTER
import subprocess
import tempfile
import time
import functools
import operator
from dmap.lower import Lower
from dmap.ir import IR, MuOp
from dmap.ops import Fusion

class C_IR:
    def __init__(self, lowered: Lower) -> None:
        self.ast = [self._transpile(branch) for branch in deepcopy(lowered.ast)]
        self.names = deepcopy(lowered.names)
        self._tokens = deepcopy(lowered.tokens)


    def _transpile(self, ast: list[IR]) -> list[IR]:
        for ir in ast:
            if ir.op == MuOp.N_D or ir.op == MuOp.N_R:
                ir.op = MuOp.LOOP

        return ast


    def to_code(self, branch: list[IR]) -> None:
        name_index = self.ast.index(branch)

        headers = ["#include <stdlib.h>\n"]
        macros = []
        main = [f"void {self.names[name_index]}("]
        indent_level = 0
        store = ""

        bin_set = {MuOp.ADD, MuOp.SUB, MuOp.MUL, MuOp.DIV}
        bin_map = {MuOp.ADD: "+", MuOp.SUB: "-", MuOp.MUL: "*", MuOp.DIV: "/"}
        bin_l_store = ""
        bin_r_store = ""

        red_set = {MuOp.SUM, MuOp.MAX, MuOp.MIN}

        un_set = {MuOp.EXP, MuOp.LOG}

        load_store = []

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
            elif ir.op in bin_set:
                if (ir.deps[0].op is MuOp.LOAD) and (ir.deps[1].op is MuOp.LOAD):
                    temp_bin = f"{load_store[0]} {bin_map[ir.op]} {load_store[1]}"
                    load_store = [temp_bin]
                elif len(bin_l_store) == 0:
                    bin_l_store = f"({ir.deps[0].value} {bin_map[ir.op]} {ir.deps[1].value})"
                elif len(bin_r_store) == 0:
                    bin_r_store = f"({ir.deps[0].value} {bin_map[ir.op]} {ir.deps[1].value})"
                else:
                    bin_l_store += f" {bin_map[ir.op]} {bin_r_store}"
                    bin_r_store = ""
            elif ir.op is MuOp.TEMP:
                main.append(("\t" * indent_level) + f"{ir.dtype} {ir.value} = {ir.deps[0].value};\n")
            elif ir.op is MuOp.LOAD:
                if len(ir.deps) == 1:
                    load_store.append(f"(*({ir.value}))")
                else:
                    load_store.append(f"(*({ir.value} + {bin_l_store}))")
                    bin_l_store = ""
            elif ir.op is MuOp.STORE:
                if ir.deps[1].op is MuOp.SUM:
                    eq_link = "+="
                else:
                    eq_link = "="
                if ir.deps[0].op is MuOp.TEMP:
                    main.append(("\t" * indent_level) + f"{ir.value} {eq_link} {load_store[-1]};\n")
                    load_store.pop(-1)
                    load_store.append(ir.value)
                else:
                    main.append(("\t" * indent_level) + f"{load_store[-1]} {eq_link} {load_store[-2]};\n")
                    load_store.pop(-1)
                    load_store.pop(-1)
            elif ir.op in red_set:
                if ir.op is MuOp.SUM:
                    continue
                else:
                    if "#include <math.h>\n" not in headers:
                        headers.append("#include <math.h>\n")
                    if ir.op is MuOp.MAX:
                        macros.append("#define MAX(x, y) (((x) > (y)) ? (x) : (y))\n")
                        load_store[-1] = f"MAX({ir.deps[0].value}, {load_store[-1]})"
                    else:
                        macros.append("#define MIN(x, y) (((x) < (y)) ? (x) : (y))\n")
                        load_store[-1] = f"MIN({ir.deps[0].value}, {load_store[-1]})"
            elif ir.op in un_set:
                if "#include <math.h>\n" not in headers:
                    headers.append("#include <math.h>\n")
                if ir.op is MuOp.EXP:
                    load_store[-1] = f"expf({load_store[-1]})"
                elif ir.op is MuOp.LOG:
                    load_store[-1] = f"logf({load_store[-1]})"
            elif ir.op is MuOp.LOOP:
                main.append(("\t" * indent_level) + f"for (int {ir.value} = {ir.deps[0].value}; {ir.value} < {ir.deps[1].value}; {ir.value}++)" + " {\n")
                indent_level += 1
            elif ir.op is MuOp.END:
                indent_level -= 1
                main.append(("\t" * indent_level) + "}\n")
        
        main.append("}")

        return "".join(headers + macros + main)


    def bench(self, branch, iters = 10) -> None:
        index = self.ast.index(branch)

        name = self.names[index]
        code = self.to_code(branch)
        token = self._tokens[index]
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