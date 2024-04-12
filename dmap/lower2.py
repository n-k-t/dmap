from typing import Callable
from dmap.tensor import Op, Tensor
from dmap.ops import Binary, Fusion, Memory, Movement, Reduce, Unary
from dmap.ir import IR, MuOp

# In order to optimize, think that we have three primary options: tesellate, unroll, and permute
#### these would both apply to the global dimensions, and we just have to figure out how to map these 
#### back to the operations and their tensors.
#### NOTE: Every time an optimization is run on an operation, we rerun the entire lowering process.
# NOTE: We could also add in keyhole optimizations later that would attempt to fuse certain opterations
#### together if possible? May not need to, though as the fusion operation is technically keyhole and would 
#### do the same thing.
# NOTE: We can check the optimizations at the apply restrictions step to make sure that everything is working as expected.
#### Order: tesselate -> unroll -> permute

# Should verify the opts first, if they fail then raise an error?
# TODO: Also need to account for if there are no optimizations applied. Need to verify singular ones and then also make sure 
# it runs as it normally does.
#### Just check if any opts were applied, if so then pass to verify function that will check all of them.

# Opts are verified, then we render the dims, set them to reduce or not, then we apply tesselation on the global dim tracker
# and insert into the ast immediately after the split dim, pair the two together in some fashion in global dim, and then 
# have to have some notation for the unroll. Lastly, we apply the permute if there is one.
# TODO: Tesselation by a factor of x!
# Instead, maybe check dims during the get global dims function...
#### TODO: Have to rework it and move out the change/move the reduce so it is inside of there/
#### Op checks just happen inside of the global dims?

# NOTE: KEY -> permute just needs to apply to the IR order, we can keep everything else as is. We can check the permute 
# after the dims have been mapped and are placed in the correct spots.
class Lower:
    def __init__(self, head: Tensor, fuse: bool = False) -> None:
        self.tokens = self._traverse(head, fuse)
        self.names = []
        self._op_match: dict[Binary|Reduce|Unary, Callable] = {
                            Binary.ADD: lambda ast, sym_tab: self._render_binary(MuOp.ADD, ast, sym_tab), 
                            Binary.SUB: lambda ast, sym_tab: self._render_binary(MuOp.SUB, ast, sym_tab), 
                            Binary.MUL: lambda ast, sym_tab: self._render_binary(MuOp.MUL, ast, sym_tab), 
                            Binary.DIV: lambda ast, sym_tab: self._render_binary(MuOp.DIV, ast, sym_tab), 
                            Reduce.SUM: lambda ast, sym_tab: self._render_reduce(MuOp.SUM, ast, sym_tab), 
                            Reduce.MAX: lambda ast, sym_tab: self._render_reduce(MuOp.MAX, ast, sym_tab), 
                            Reduce.MIN: lambda ast, sym_tab: self._render_reduce(MuOp.MIN, ast, sym_tab), 
                            Unary.EXP: lambda ast, sym_tab: self._render_unary(MuOp.EXP, ast, sym_tab), 
                            Unary.LOG: lambda ast, sym_tab: self._render_unary(MuOp.LOG, ast, sym_tab)
                        }
        self.ast = [self._lower(token) for token in self.tokens]


    def _traverse(self, tensor: Tensor, fuse: bool) -> list[Op]:
        def top_sort(tensor, visited, stack) -> list[Tensor]:
            visited.add(tensor)
            for parent in tensor.parents:
                if parent not in visited:
                    top_sort(parent, visited, stack)
            if tensor.op not in visited:
                visited.add(tensor.op)
                stack.append(tensor.op)
            return stack
        
        if fuse:
            return self._apply_restrictions(self._fuse(top_sort(tensor, set(), [])))
        else:
            return self._apply_restrictions(top_sort(tensor, set(), []))


    def _fuse(self, ops: list[Op]) -> list[Op]:
        index: int = 0

        while index < (len(ops) - 1):
            if isinstance(ops[index].op, Binary) and isinstance(ops[index + 1].op, Reduce) and (ops[index].t_out in ops[index + 1].t_in):
                op_holder = Op(Fusion.ELE_RED)
                op_holder.num_flop = ops[index].num_flop + ops[index + 1].num_flop
                op_holder.fus_ops = ops[index: index + 2]
                op_holder.axis = ops[index + 1].axis
                ops.pop(index)
                ops[index] = op_holder
            index += 1
        
        return ops


    def _apply_restrictions(self, token_stream: list[Op]) -> list[Op]:        
        return [token for token in token_stream if (not isinstance(token.op, Memory)) and (not isinstance(token.op, Movement))]
    

    def _lower(self, token: Op) -> list[IR]:
        temp_name: str = self._gen_kernel_name(token) + "_v"
        repetitions: int = 0
        for name in self.names:
            if temp_name in name:
                repetitions += 1
        
        self.names.append(temp_name + f"{repetitions}")

        queue: list[Op] = self._enqueue(token)

        ast: list[IR] = []
        ctx: dict[str, list[Tensor]] = {"L": [], "S": []}
        sym_tab: dict[Tensor|str, IR|list[IR]] = {}

        self._define_ctx(queue, ast, ctx, sym_tab)

        # TODO: Now we track the axis for reduction, so we can pass the axis inside with the opts and move the make_and_move
        self._get_g_dims(ast, ctx, sym_tab)

        for op in queue:
            if isinstance(op.op, Reduce):
                self._make_and_move_reduce_axis(ast, op.axis, sym_tab["g_dims"])

            sym_tab["in_t_point"] = []

            for parent in op.t_in:
                sym_tab["in_t_point"].append(self._index_in(parent, ast, sym_tab))
            
            # NOTE: Could we pass around an unroll factor that allows us to track what exactly was unrolled.
            #### in_t_point will have first n/2 entries as input 1 and second n/2 entries as input 2. Then, if we know
            #### the factor, we can take the [ind] and [ind + n/2] entries (for a binary op ex.) where ind runs from 0 to n/2
            #### AKA, we can index in and render the loads in a loop setting.
            # TODO: I think that I may need to figure out how to map to the specific N_R if we have multiple reduces happening;
            # otherwise, all temporary variables will always be mapped to the outer most reduce.
            # I might be able to do this by tracking back through the global dimensions. But again, I have to figure out how I 
            # will deal with the tesselations because you can't index into a nested structure so easily.
            self._op_match[op.op](ast, sym_tab)

            if isinstance(op.op, Reduce):
                sym_tab["out_t_point"] = self._index_in(op.t_out, ast, sym_tab, op.axis)
            else:
                sym_tab["out_t_point"] = self._index_in(op.t_out, ast, sym_tab)

            ast.append(IR(MuOp.STORE, sym_tab["out_t_point"].dtype, sym_tab["out_t_point"].value, [sym_tab["out_t_point"], sym_tab["link_point"]]))

        self._purge_unused(ast)

        self._end_dims(ast, sym_tab)

        return ast


    def _gen_kernel_name(self, op: Op) -> str:
        if isinstance(op.op, Unary) or isinstance(op.op, Binary):
            return "_".join(["E"] + [str(dim) for dim in op.t_in[0].view] + ["to"] + [str(dim) for dim in op.t_out.view])
        elif isinstance(op.op, Reduce):
            return "_".join(["R"] + [str(dim) for dim in op.t_in[0].view] + ["to"] + [str(dim) for dim in op.t_out.view])
        elif isinstance(op.op, Fusion):
            return "_".join(["F"] + [str(dim) for dim in op.fus_ops[0].t_in[0].view] + ["to"] + [str(dim) for dim in op.fus_ops[-1].t_out.view])


    def _enqueue(self, op: Op) -> list[Op]:
        if isinstance(op.op, Fusion):
            return op.fus_ops
        else:
            return [op]


    def _define_ctx(self, queue: list[Op], ast: list[IR], ctx: dict[str, list[Tensor]], sym_tab: dict[Tensor|str, IR|list[IR]]) -> None:
        tensor_count: int = 0

        for op in queue:
            for parent in op.t_in:
                if parent in ctx["S"]:
                    sym_tab[parent].value = f"in_{tensor_count}"
                    ctx["L"].append(parent)
                else:
                    sym_tab[parent] = IR(MuOp.ARG, parent.dtype + "*", f"in_{tensor_count}", [])
                    ast.append(sym_tab[parent])
                    ctx["L"].append(parent)
                tensor_count += 1
            
            sym_tab[op.t_out] = IR(MuOp.ARG, op.t_out.dtype + "*", "out", [])
            ast.append(sym_tab[op.t_out])
            ctx["S"].append(op.t_out)


    def _get_g_dims(self, ast: list[IR], ctx: dict[str, list[Tensor]], sym_tab: dict[Tensor|str, IR|list[IR]]) -> None:
        sym_tab["g_dims"] = []

        if "0" not in sym_tab:
            sym_tab["0"] = IR(MuOp.CONST, "int", "0", [])
            ast.append(sym_tab["0"])

        for num, dim in enumerate(ctx["L"][0].view):
            if str(dim) not in sym_tab:
                sym_tab[str(dim)] = IR(MuOp.CONST, "int", str(dim), [])
                ast.append(sym_tab[str(dim)])
            
            sym_tab["g_dims"].append(IR(MuOp.N_D, "", f"axis_{num}", [sym_tab["0"], sym_tab[str(dim)]]))
            ast.append(sym_tab["g_dims"][-1])


    def _make_and_move_reduce_axis(self, ast: list[IR], index: int, dims: list[IR]):
        dims[index].op = MuOp.N_R
        dims[index].value = "r_" + dims[index].value
        red_ax_ind = ast.index(dims[index])
        inner_ax_ind = max([ast.index(dim) for dim in dims])
        if red_ax_ind != inner_ax_ind:
            holder = ast[red_ax_ind]
            ast[red_ax_ind] = ast[inner_ax_ind]
            ast[inner_ax_ind] = holder


    def _index_in(self, tensor: Tensor, ast: list[IR], sym_tab: dict[Tensor|str, IR|list[IR]], red_axis: int = -1) -> IR:
        shape = tensor.view
        g_dims = sym_tab["g_dims"]

        dim_map = self._map_dims(g_dims, shape, red_axis)

        if len(dim_map) == 0:
            ast.append(IR(MuOp.LOAD, tensor.dtype, sym_tab[tensor].value, [sym_tab[tensor]]))
            return ast[-1]
        
        stride = tensor.stride

        # NOTE: Think I'll need to condition inside if the mapped index is tesselated
        # I may also want to remove the index tracker and instead create a back tracker -> Optional[IR]
        # How will I handle unrolling? Especially since you could unroll a tesselated dimension.
        # Does the unrolling happen at a later stage? I.e. how do you unroll two operands and track them all
        for index, dim in enumerate(dim_map):
            if str(stride[index]) not in sym_tab:
                sym_tab[str(stride[index])] = IR(MuOp.CONST, "int", stride[index], [])
                ast.append(sym_tab[str(stride[index])])
            temp_op: IR = IR(MuOp.MUL, "int", "", [g_dims[dim], sym_tab[str(stride[index])]])
            ast.append(temp_op)
            if index != 0:
                temp_op = IR(MuOp.ADD, "int", "", [previous_term, temp_op])
                ast.append(temp_op)
            previous_term = temp_op

        ast.append(IR(MuOp.LOAD, tensor.dtype, sym_tab[tensor].value, [sym_tab[tensor], ast[-1]]))

        return ast[-1]


    def _map_dims(self, g_dims: list[IR], shape: list[int], red_axis: int) -> list[int]:
        if len(shape) == 1 and (shape[0] != 1):
            return [ind for ind, _ in enumerate(g_dims) if ind != red_axis]
        else:
            map_g = [ind for ind, _ in enumerate(g_dims) if ind != red_axis]
            map_s = [ind for ind, dim in enumerate(shape) if dim != 1]
            return[map_g[ind] for ind in map_s]
    

    def _render_binary(self, op: MuOp, ast: list[IR], sym_tab: dict[Tensor|str, IR|list[IR]]) -> None:
        tensors: list[IR] = sym_tab["in_t_point"]

        sym_tab["link_point"] = IR(op, tensors[0].dtype, "", tensors)

        ast.append(sym_tab["link_point"])


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


    def _render_reduce(self, op: MuOp, ast: list[IR], sym_tab: dict[Tensor|str, IR|list[IR]]) -> None:
        match_init_val: dict[MuOp, str] = {MuOp.SUM: "0", MuOp.MAX: "-INFINITY", MuOp.MIN: "INFINITY"}
        red_init_val: str = match_init_val[op]
        tensors: list[IR] = sym_tab["in_t_point"]

        deps: set[IR] = self._find_ir_children(tensors[0])

        red_inds = [ast.index(dep) for dep in deps if dep.op is MuOp.N_R]

        if len(red_inds) == 0:
            temp_index = max([ast.index(op) for op in ast if op.op is MuOp.N_R])
        else:
            temp_index: int = max(red_inds)

        if red_init_val not in sym_tab:
            sym_tab[red_init_val] = IR(MuOp.CONST, "int", red_init_val, [])
            ast.insert(temp_index, sym_tab[red_init_val])
            temp_index += 1

        temp_count: int = 0
        for ir in ast:
            if ir.op is MuOp.TEMP:
                temp_count += 1

        sym_tab["link_point"] = IR(MuOp.TEMP, tensors[0].dtype, f"temp_{temp_count}", [sym_tab[red_init_val]])
        ast.insert(temp_index, sym_tab["link_point"])

        if op is MuOp.SUM:
            ast.append(IR(op, tensors[0].dtype, "", tensors))
        else:
            ast.append(IR(op, tensors[0].dtype, "", [sym_tab["link_point"]] + tensors))

        ast.append(IR(MuOp.STORE, tensors[0].dtype, sym_tab["link_point"].value, [sym_tab["link_point"], ast[-1]]))


    def _render_unary(self, op: MuOp, ast: list[IR], sym_tab: dict[Tensor|str, IR|list[IR]]) -> None:
        tensors: list[IR] = sym_tab["in_t_point"]

        sym_tab["link_point"] = IR(op, tensors[0].dtype, "", tensors)

        ast.append(sym_tab["link_point"])


    def _purge_unused(self, ast: list[IR]) -> None:
        used = set()

        for ir in ast:
            if ir.op is MuOp.STORE:
                used.add(ir)

            for dep in ir.deps:
                if dep not in used:
                    used.add(dep)

        for ind in range(len(ast) - 1, -1, -1):
            if ast[ind] not in used:
                ast.pop(ind)


    def _end_dims(self, ast: list[IR], sym_tab: dict[Tensor|str, IR|list[IR]]) -> None:
        for axis in sym_tab["g_dims"]:
            if axis not in ast:
                continue

            farthest_loop_dep = [ast.index(ir) for ir in self._find_ir_parents(axis, ast)]
            end_index = max(farthest_loop_dep) + 1
            ast.insert(end_index, IR(MuOp.END, "", axis.value, [axis]))