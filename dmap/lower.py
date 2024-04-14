from typing import Callable
from itertools import product
from dmap.tensor import Op, Tensor
from dmap.ops import Binary, Fusion, Reduce, Unary
from dmap.ir import IR, MuOp


class Lower:
    def __init__(self, token: Op, tesselate: list[list[int]] = [], unroll: list[int] = [], permute: list[int] = []) -> None:
        self._op_match: dict[Binary|Reduce|Unary, Callable] = {
                            Binary.ADD: lambda t_inds, offset, op_link, _, __, ___, ____, _____, ______, _______: self._render_binary(MuOp.ADD, t_inds, offset, op_link), 
                            Binary.MUL: lambda t_inds, offset, op_link, _, __, ___, ____, _____, ______, _______: self._render_binary(MuOp.MUL, t_inds, offset, op_link), 
                            Binary.DIV: lambda t_inds, offset, op_link, _, __, ___, ____, _____, ______, _______: self._render_binary(MuOp.DIV, t_inds, offset, op_link), 
                            Binary.SUB: lambda t_inds, offset, op_link, _, __, ___, ____, _____, ______, _______: self._render_binary(MuOp.SUB, t_inds, offset, op_link), 
                            Reduce.SUM: lambda t_inds, _, op_link, dim_dep_in, dim_track, red_ax, n_temp, const_t, a_dim, tess_t: self._render_reduce(MuOp.SUM, t_inds, op_link, dim_dep_in, dim_track, red_ax, n_temp, const_t, a_dim, tess_t), 
                            Reduce.MAX: lambda t_inds, _, op_link, dim_dep_in, dim_track, red_ax, n_temp, const_t, a_dim, tess_t: self._render_reduce(MuOp.MAX, t_inds, op_link, dim_dep_in, dim_track, red_ax, n_temp, const_t, a_dim, tess_t), 
                            Reduce.MIN: lambda t_inds, _, op_link, dim_dep_in, dim_track, red_ax, n_temp, const_t, a_dim, tess_t: self._render_reduce(MuOp.MIN, t_inds, op_link, dim_dep_in, dim_track, red_ax, n_temp, const_t, a_dim, tess_t), 
                            Unary.EXP: lambda t_inds, _, op_link, __, ___, ____, _____, ______, _______, ________: self._render_unary(MuOp.EXP, t_inds, op_link), 
                            Unary.LOG: lambda t_inds, _, op_link, __, ___, ____, _____, ______, _______, ________: self._render_unary(MuOp.LOG, t_inds, op_link)
                        }
        self._tesselate = tesselate
        self._unroll = unroll
        self._permute = permute
        self.ast = self._lower(token)


    def _lower(self, token: Op) -> list[IR]:
        queue: list[Op] = self._enqueue(token)

        ast: list[IR] = []
        ctx: dict[str, list[Tensor]] = {"L": [], "S": []}
        tensor_tracker: dict[Tensor, IR] = {}

        self._define_ctx_and_args(queue, ast, ctx, tensor_tracker)

        self._validate_opts(self._tesselate, self._unroll, self._permute, ctx["L"][0].view, token.axis)

        dim_tracker, const_tracker, tess_tracker, unro_tracker, all_dims = self._get_g_dims(ctx)

        reduce_counter: int = 0
        reduce_axes: list[int] = token.axis
        offset_tracker: dict[Op, list[int]] = {}
        op_in_t_inds: dict[Op, list[list[IR]]] = {}
        op_out_t_inds: dict[Op, list[list[IR]]] = {}
        op_link: dict[Op, list[IR]] = {}
        dim_deps_in: dict[Op, dict[IR, list[IR]]] = {}
        dim_deps_out: dict[Op, dict[IR, list[IR]]] = {}

        all_dims = self._make_and_move_reduce_axis(reduce_axes, dim_tracker, tess_tracker, all_dims)

        for op in queue:
            offset_tracker[op] = []
            op_in_t_inds[op] = []

            for parent in op.t_in:
                self._index_in(parent, tensor_tracker, dim_tracker, tess_tracker, unro_tracker, const_tracker, offset_tracker[op], op_in_t_inds[op])

            op_link[op] = []
            dim_deps_in[op] = {}

            num_temps = 1

            if isinstance(op.op, Reduce):
                red_ax = reduce_axes[reduce_counter]

                for i in unro_tracker:
                    if i.op is MuOp.N_D:
                        num_temps *= unro_tracker[i]
            else:
                red_ax = -1
            

            self._op_match[op.op](op_in_t_inds[op], offset_tracker[op], op_link[op], dim_deps_in[op], dim_tracker, \
                                    red_ax, num_temps, const_tracker, all_dims, tess_tracker)

            op_out_t_inds[op] = []

            if isinstance(op.op, Reduce):
                dim_deps_out[op] = {}

                self._index_in(op.t_out, tensor_tracker, dim_tracker, tess_tracker, unro_tracker, \
                                const_tracker, offset_tracker[op], op_out_t_inds[op], red_ax)

                self._finish_reduce(op_out_t_inds[op], dim_deps_out[op], dim_deps_in[op], dim_tracker, tess_tracker, all_dims, red_ax)

                op_out_t_inds[op].clear()

                reduce_counter += 1
            else:
                self._index_in(op.t_out, tensor_tracker, dim_tracker, tess_tracker, unro_tracker, \
                                const_tracker, offset_tracker[op], op_out_t_inds[op])
                
                self._finish_una_bin(op_out_t_inds[op], op_link[op])

        for c in const_tracker.values():
            ast.append(c)

        for dim in all_dims:
            for op in queue:
                if dim in dim_deps_in[op]:
                    for dep in dim_deps_in[op][dim]:
                        ast.append(dep)

            if dim in unro_tracker:
                continue
            elif dim.deps[1].value != "1":
                ast.append(dim)
        
        for op in queue:
            for t_in in op_in_t_inds[op]:
                for ir in t_in:
                    ast.append(ir)

            for o_l in op_link[op]:
                ast.append(o_l)

            for t_out in op_out_t_inds[op]:
                for ir in t_out:
                    ast.append(ir)
        
        for dim in reversed(all_dims):
            if (dim.deps[1].value != "1") and (dim not in unro_tracker):
                ast.append(IR(MuOp.END, "", dim.value, [dim]))
            
            for op in queue:
                if op in dim_deps_out:
                    if dim in dim_deps_out[op]:
                        for dep in dim_deps_out[op][dim]:
                            ast.append(dep)

        return ast


    def _enqueue(self, op: Op) -> list[Op]:
        if isinstance(op.op, Fusion):
            return op.fus_ops
        else:
            return [op]


    def _define_ctx_and_args(self, queue: list[Op], ast: list[IR], ctx: dict[str, list[Tensor]], tensor_tracker: dict[Tensor, IR]) -> None:
        tensor_count: int = 0

        for op in queue:
            for parent in op.t_in:
                if parent in ctx["S"]:
                    tensor_tracker[parent].value = f"in_{tensor_count}"
                    ctx["L"].append(parent)
                else:
                    tensor_tracker[parent] = IR(MuOp.ARG, parent.dtype + "*", f"in_{tensor_count}", [])
                    ast.append(tensor_tracker[parent])
                    ctx["L"].append(parent)
                tensor_count += 1
            
            tensor_tracker[op.t_out] = IR(MuOp.ARG, op.t_out.dtype + "*", "out", [])
            ast.append(tensor_tracker[op.t_out])
            ctx["S"].append(op.t_out)


    def _validate_opts(self, tesselate: list[list[int]], unroll: list[int], permute: list[int], global_shape: list[int], red_axes: list[int]) -> None:
        dim_set = set()
        num_dims = len(global_shape)

        for i in tesselate:
            if (i[0] >= 0) and (i[0] < len(global_shape)):
                num_dims += 1
            else:
                raise ValueError("You can't tesselate a dimension index outside of the defined range.")
            if i[0] in dim_set:
                raise ValueError("You can't tesselate the same dimension more than once.")
            else:
                dim_set.add(i[0])
            if (i[1] == 1) or (i[1] == global_shape[i[0]]):
                raise ValueError("Tesselating by 1 or the dimension size results in no change, so it is not allowed.")
            if i[1] <= 0:
                raise ValueError("You can't tesselate by a factor <= 0.")
            if global_shape[i[0]] % i[1] != 0:
                raise ValueError("The tesselation factor is not a divisor of the specified dimension.")

        dim_set.clear()

        for d in unroll:
            if not ((d >= 0) and (d < num_dims)):
                raise ValueError("You can't unroll a dimension index outside the defined range.")
            if d in dim_set:
                raise ValueError("You can't unroll the same dimension more than once.")
            else:
                dim_set.add(d)

        if len(permute) > 0:
            if len(permute) > num_dims:
                raise ValueError("You can't permute more dimensions than provided to the operations.")
            
            if len(permute) < num_dims:
                raise ValueError("You can't permute fewer dimensions than provided to the operations.")
            
            dim_set.clear()

            for i in permute:
                if i in dim_set:
                    raise ValueError("You can't have duplicate dimensions in your permutation.")
                else:
                    dim_set.add(i)

            if set(permute) != set([i for i in range(num_dims)]):
                raise ValueError("Your permutation does not include all dimensions in the operation's dimensional range.")

            if len(red_axes) > 0:
                offset = 0
                dim_set.clear()
                tes_set = sorted(set([i[0] for i in tesselate]))
                for i in tes_set:
                    if i in red_axes:
                        dim_set.add(i + offset)
                        dim_set.add(i + offset + 1)
                    offset += 1
                if len(dim_set) > 0:
                    if dim_set != set(permute[-len(dim_set):len(permute)]):
                        raise ValueError("The reduce dimensions aren't the most nested (last) in the permutation given.")
                elif set(red_axes) != set(permute[-len(red_axes):len(permute)]):
                    raise ValueError("The reduce dimensions aren't the most nested (last) in the permutation given.")


    def _check_consts(self, val: int, const_tracker: dict[str, IR]) -> None:
        if str(val) not in const_tracker:
            const_tracker[str(val)] = IR(MuOp.CONST, "int", str(val), [])


    def _get_g_dims(self, ctx: dict[str, list[Tensor]]) -> tuple[list[IR], dict[str, IR], dict[IR, tuple[int, IR]], dict[IR, int], list[IR]]:
        dim_tracker: list[IR] = []
        const_tracker: dict[str, IR] = {}
        tess_tracker: dict[IR, tuple[int, IR]] = {}
        unro_tracker: dict[IR, int] = {}

        tess_dims: list[int] = [i[0] for i in self._tesselate]
        tess_factors: list[int] = [i[1] for i in self._tesselate]
        all_dims: list[IR] = []

        self._check_consts(0, const_tracker)

        for num, dim in enumerate(ctx["L"][0].view):
            self._check_consts(dim, const_tracker)
            
            dim_tracker.append(IR(MuOp.N_D, "", f"axis_{num}", [const_tracker["0"], const_tracker[str(dim)]]))

            all_dims.append(dim_tracker[num])

            if num in tess_dims:
                tess_factor = tess_factors[tess_dims.index(num)]
                scaled_val = int(dim / tess_factor)

                self._check_consts(tess_factor, const_tracker)
                self._check_consts(scaled_val, const_tracker)

                dim_tracker[num].deps[1] = const_tracker[str(scaled_val)]

                temp_ir = IR(MuOp.N_D, "", "t_" + dim_tracker[num].value, [const_tracker["0"], const_tracker[str(tess_factor)]])

                tess_tracker[dim_tracker[num]] = (tess_factor, temp_ir)
                all_dims.append(temp_ir)

        for dim in self._unroll:
            unro_tracker[all_dims[dim]] = int(all_dims[dim].deps[1].value)

        return (dim_tracker, const_tracker, tess_tracker, unro_tracker, all_dims)


    def _make_and_move_reduce_axis(self, indices: list[int], dim_tracker: list[IR], tess_tracker: dict[IR, tuple[int, IR]], all_dims: list[IR]) -> list[IR]:
        red_axes = []
        
        for ind in indices:
            dim_tracker[ind].op = MuOp.N_R
            dim_tracker[ind].value = "r_" + dim_tracker[ind].value

            red_axes.append(all_dims.index(dim_tracker[ind]))

            if dim_tracker[ind] in tess_tracker:
                tess_tracker[dim_tracker[ind]][1].op = MuOp.N_R
                tess_tracker[dim_tracker[ind]][1].value = "r_" + tess_tracker[dim_tracker[ind]][1].value

                red_axes.append(red_axes[-1] + 1)

        if len(self._permute) > 0:
            temp_order = [all_dims[i] for i in self._permute]
            all_dims = temp_order
        else:
            length_set = set([i for i in range(len(all_dims))])
            non_red_axes = length_set.symmetric_difference(red_axes)
            all_axes = list(non_red_axes) + red_axes

            temp_order = [all_dims[i] for i in all_axes]
            all_dims = temp_order

        return all_dims


    def _index_in(
                    self, 
                    tensor: Tensor, 
                    tensor_tracker: dict[Tensor, IR], 
                    dim_tracker: list[IR], 
                    tess_tracker: dict[IR, tuple[int, IR]], 
                    unro_tracker: dict[IR, int], 
                    const_tracker: dict[str, IR], 
                    offset_tracker: list[int], 
                    op_t_inds: list[list[IR]], 
                    red_ind: int = -1
                ) -> None:
        shape = tensor.view

        dim_map = self._map_dims(dim_tracker, shape, red_ind)

        if len(dim_map) == 0:
            op_t_inds.append([IR(MuOp.LOAD, tensor.dtype, tensor_tracker[tensor].value, [tensor_tracker[tensor]])])
            offset_tracker.append(1)
            return

        opt_dim_vals = []
        new_stride = []
        temp_offset = 1

        for dim, s in zip(dim_map, tensor.stride):
            if dim_tracker[dim] in unro_tracker:
                temp_offset *= unro_tracker[dim_tracker[dim]]
                opt_dim_vals.append(range(unro_tracker[dim_tracker[dim]]))
            else:
                opt_dim_vals.append([dim_tracker[dim]])
            if dim_tracker[dim] in tess_tracker:
                scaled_stride = s * tess_tracker[dim_tracker[dim]][0]
                new_stride.append(scaled_stride)
                self._check_consts(scaled_stride, const_tracker)

                new_stride.append(s)
                self._check_consts(s, const_tracker)

                if tess_tracker[dim_tracker[dim]][1] in unro_tracker:
                    temp_offset *= unro_tracker[tess_tracker[dim_tracker[dim]][1]]
                    opt_dim_vals.append(range(unro_tracker[tess_tracker[dim_tracker[dim]][1]]))
                else:
                    opt_dim_vals.append([tess_tracker[dim_tracker[dim]][1]])
            else:
                new_stride.append(s)
                self._check_consts(s, const_tracker)
        
        offset_tracker.append(temp_offset)

        for mapping in product(*opt_dim_vals):
            temp_index_list = []
            previous_term = None

            for index, stride in enumerate(new_stride):
                if isinstance(mapping[index], IR):
                    temp_op: IR = IR(MuOp.MUL, "int", "", [mapping[index], const_tracker[str(stride)]])
                    temp_index_list.append(temp_op)
                else:
                    single_stride = mapping[index] * stride
                    self._check_consts(single_stride, const_tracker)
                    temp_op = const_tracker[str(single_stride)]

                if index != 0:
                    temp_op = IR(MuOp.ADD, "int", "", [previous_term, temp_op])
                    temp_index_list.append(temp_op)
                
                previous_term = temp_op

            temp_index_list.append(IR(MuOp.LOAD, tensor.dtype, tensor_tracker[tensor].value, [tensor_tracker[tensor], previous_term]))

            op_t_inds.append(temp_index_list)


    def _map_dims(self, dim_tracker: list[IR], shape: list[int], red_ind: int) -> list[int]:
        if len(shape) == 1 and (shape[0] != 1):
            return [ind for ind, _ in enumerate(dim_tracker) if ind != red_ind]
        else:
            map_g = [ind for ind, _ in enumerate(dim_tracker) if ind != red_ind]
            map_s = [ind for ind, dim in enumerate(shape) if dim != 1]
            return[map_g[ind] for ind in map_s]
    

    def _render_binary(self, op: MuOp, op_in_t_inds: list[list[IR]], offset_tracker: list[int], op_link: list[IR]) -> None:
        last_ele_ind = len(op_in_t_inds[0]) - 1
        data_type = op_in_t_inds[0][last_ele_ind].dtype

        for i in range(offset_tracker[0]):
            op_link.append(IR(op, data_type, "", [op_in_t_inds[i][last_ele_ind], op_in_t_inds[i + offset_tracker[0]][last_ele_ind]]))


    def _render_unary(self, op: MuOp, op_in_t_inds: list[list[IR]], op_link: list[IR]) -> None:
        last_ele_ind = len(op_in_t_inds[0]) - 1
        data_type = op_in_t_inds[0][last_ele_ind].dtype

        for i in op_in_t_inds:
            op_link.append(IR(op, data_type, "", [i[last_ele_ind]]))


    def _render_reduce(
                        self, 
                        op: MuOp, 
                        op_in_t_inds: list[list[IR]], 
                        op_link: list[IR], 
                        dim_deps_in: dict[IR, list[IR]], 
                        dim_tracker: list[IR], 
                        red_ax: int, 
                        num_temp: int, 
                        const_tracker: dict[str, IR], 
                        all_dims: list[IR], 
                        tess_tracker: dict[IR, tuple[int, IR]], 
                    ) -> None:
        match_init_val: dict[MuOp, str] = {MuOp.SUM: "0", MuOp.MAX: "-INFINITY", MuOp.MIN: "INFINITY"}
        init_val = match_init_val[op]
        self._check_consts(init_val, const_tracker)

        last_ele_ind = len(op_in_t_inds[0]) - 1
        data_type = op_in_t_inds[0][last_ele_ind].dtype

        red_dim: IR = dim_tracker[red_ax]
        if red_dim in tess_tracker:
            min_ind = min([all_dims.index(red_dim), all_dims.index(tess_tracker[red_dim][1])])
            red_dim = all_dims[min_ind]
        dim_deps_in[red_dim] = []

        for i in range(num_temp):
            dim_deps_in[red_dim].append(IR(MuOp.TEMP, data_type, f"temp_{i}", [const_tracker[str(init_val)]]))

        if op is MuOp.SUM:
            for i in range(num_temp):
                for j in op_in_t_inds:
                    temp_ir = IR(op, data_type, "", [j[last_ele_ind]])
                    op_link.append(temp_ir)
                    op_link.append(IR(MuOp.STORE, data_type, dim_deps_in[red_dim][i].value, [dim_deps_in[red_dim][i], temp_ir]))
        else:
            for i in range(num_temp):
                for j in op_in_t_inds:
                    temp_ir = IR(op, data_type, "", [dim_deps_in[red_dim][i], j[last_ele_ind]])
                    op_link.append(temp_ir)
                    op_link.append(IR(MuOp.STORE, data_type, dim_deps_in[red_dim][i].value, [dim_deps_in[red_dim][i], temp_ir]))


    def _finish_reduce(
                        self, 
                        op_out_t_inds: list[list[IR]], 
                        dim_deps_out: list[list[IR]], 
                        dim_deps_in: list[list[IR]], 
                        dim_tracker: list[IR], 
                        tess_tracker: dict[IR, tuple[int, IR]], 
                        all_dims: list[IR], 
                        red_ax: int
                    ) -> None:
        last_ele_ind = len(op_out_t_inds[0]) - 1
        data_type = op_out_t_inds[0][last_ele_ind].dtype

        red_dim: IR = dim_tracker[red_ax]
        if red_dim in tess_tracker:
            min_ind = min([all_dims.index(red_dim), all_dims.index(tess_tracker[red_dim][1])])
            red_dim = all_dims[min_ind]
        dim_deps_out[red_dim] = []

        for out_t, in_dep in zip(op_out_t_inds, dim_deps_in[red_dim]):
            for ir in out_t:
                dim_deps_out[red_dim].append(ir)

            dim_deps_out[red_dim].append(IR(MuOp.STORE, data_type, out_t[last_ele_ind].value, [out_t[last_ele_ind], in_dep]))


    def _finish_una_bin(self, op_out_t_inds: list[list[IR]], op_link: list[IR]) -> None:
        last_ele_ind = len(op_out_t_inds[0]) - 1
        data_type = op_out_t_inds[0][last_ele_ind].dtype

        for num, t_out in enumerate(op_out_t_inds):
            t_out.append(IR(MuOp.STORE, data_type, t_out[last_ele_ind].value, [t_out[last_ele_ind], op_link[num]]))


    def _find_chained_ast_interactions(self, ir: IR, ast: list[IR]) -> set[IR]:
        parent_set: set[IR] = set([ir])

        delta_len: int = len(parent_set)

        while delta_len != 0:
            start_len: int = len(parent_set)
            for ir in ast:
                if parent_set.intersection(ir.deps):
                    parent_set.add(ir)
            delta_len = len(parent_set) - start_len

        return parent_set


    def _find_direct_deps_chain(self, ir: IR) -> set[IR]:
        return set.union(set(ir.deps), *[self._find_direct_deps_chain(dep) for dep in ir.deps])


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

            farthest_loop_dep = [ast.index(ir) for ir in self._find_chained_ast_interactions(axis, ast)]
            end_index = max(farthest_loop_dep) + 1
            ast.insert(end_index, IR(MuOp.END, "", axis.value, [axis]))