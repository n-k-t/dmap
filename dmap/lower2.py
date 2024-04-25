import functools, operator
from typing import Callable
from itertools import product
from dmap.tensor import Op, Tensor
from dmap.ops import Binary, Fusion, Reduce, Unary
from dmap.ir import IR, MuOp


class Opt:
    def __init__(self, tesselate = [], permute = [], unroll = [], globals = [], locals = [], threads_per_block = 1024, shared_mem = None) -> None:
        self.tesselate: list[tuple[int, int]] = tesselate
        self.permute: list[int] = permute
        self.unroll: list[int] = unroll
        self.globals: list[int] = globals
        self.locals: list[int] = locals
        self.threads_per_block: int = threads_per_block
        self.shared_mem: int|None = shared_mem


class Lower:
    def __init__(self, token: Op, opt: Opt|None = None) -> None:
        self.blocks_per_grid: int|None = None
        self.threads_per_block: int|None = None
        self._op_match: dict[Binary|Reduce|Unary, Callable] = {
                            Binary.ADD: lambda t_inds, offset, op_link, _, __, ___, ____, _____: self._render_binary(MuOp.ADD, t_inds, offset, op_link), 
                            Binary.MUL: lambda t_inds, offset, op_link, _, __, ___, ____, _____: self._render_binary(MuOp.MUL, t_inds, offset, op_link), 
                            Binary.DIV: lambda t_inds, offset, op_link, _, __, ___, ____, _____: self._render_binary(MuOp.DIV, t_inds, offset, op_link), 
                            Binary.SUB: lambda t_inds, offset, op_link, _, __, ___, ____, _____: self._render_binary(MuOp.SUB, t_inds, offset, op_link), 
                            Reduce.SUM: lambda t_inds, _, op_link, dim_dep_in, red_ax, r_m_fac, a_dim, const_t: self._render_reduce(MuOp.ADD, t_inds, op_link, dim_dep_in, red_ax, r_m_fac, a_dim, const_t), 
                            Reduce.MAX: lambda t_inds, _, op_link, dim_dep_in, red_ax, r_m_fac, a_dim, const_t: self._render_reduce(MuOp.MAX, t_inds, op_link, dim_dep_in, red_ax, r_m_fac, a_dim, const_t), 
                            Reduce.MIN: lambda t_inds, _, op_link, dim_dep_in, red_ax, r_m_fac, a_dim, const_t: self._render_reduce(MuOp.MIN, t_inds, op_link, dim_dep_in, red_ax, r_m_fac, a_dim, const_t), 
                            Unary.EXP: lambda t_inds, _, op_link, __, ___, ____, _____, ______: self._render_unary(MuOp.EXP, t_inds, op_link), 
                            Unary.LOG: lambda t_inds, _, op_link, __, ___, ____, _____, ______: self._render_unary(MuOp.LOG, t_inds, op_link)
                        }
        self.ast = self._lower(token, opt)


    def _lower(self, token: Op, opt: Opt|None) -> list[IR]:
        queue: list[Op] = self._enqueue(token)

        global_shape = queue[0].t_in[0].view
        red_axes = token.axis

        if opt:
            self._val_opts(opt, red_axes, global_shape)

        ast: list[IR] = []
        ctx: dict[Tensor, str] = {}
        tensor_tracker: dict[Tensor, IR] = {}

        self._define_ctx_and_args(queue, ast, ctx, tensor_tracker)


        const_tracker: dict[str, IR] = {}

        all_dims, ordered_dims, stride_tracker, unro_tracker, new_global_shape, new_red_axes = self._update_and_map_dims(global_shape, red_axes, const_tracker, ctx, opt)


        global_map, local_map = self._map_parallel_dims(opt.globals, opt.locals, all_dims, opt.threads_per_block, const_tracker)


        offset_tracker: dict[Op, list[int]] = {}
        op_in_t_inds: dict[Op, list[list[IR]]] = {}
        op_out_t_inds: dict[Op, list[list[IR]]] = {}
        op_link: dict[Op, list[IR]] = {}
        dim_deps_in: dict[Op, dict[IR, list[IR]]] = {}
        dim_deps_out: dict[Op, dict[IR, list[IR]]] = {}

        red_count = 0

        for op in queue:
            if isinstance(op.op, Reduce):
                store_ax: int = -1
                current_min: int = -1
                for num, i in enumerate(new_red_axes[red_count]):
                    if num == 0:
                        current_min = ordered_dims.index(all_dims[i])
                        store_ax = i
                    elif ordered_dims.index(all_dims[i]) < current_min:
                        current_min = ordered_dims.index(all_dims[i])
                        store_ax = i
                min_red_ax = store_ax
                all_red_axes = new_red_axes[red_count]
                red_count += 1
            else:
                min_red_ax = None
                all_red_axes = None


            offset_tracker[op] = []
            op_in_t_inds[op] = []

            for parent in op.t_in:
                self._index_in(parent, tensor_tracker, stride_tracker, all_dims, unro_tracker, const_tracker, offset_tracker[op], op_in_t_inds[op])

            op_link[op] = []
            dim_deps_in[op] = {}

            if isinstance(op.op, Reduce):
                red_map_factor = 1
                for dim in all_red_axes:
                    if all_dims[dim] in unro_tracker:
                        red_map_factor *= unro_tracker[all_dims[dim]]
            else:
                red_map_factor = 0


            self._op_match[op.op](op_in_t_inds[op], offset_tracker[op], op_link[op], dim_deps_in[op],
                                    min_red_ax, red_map_factor, all_dims, const_tracker)

            op_out_t_inds[op] = []

            if isinstance(op.op, Reduce):
                dim_deps_out[op] = {}

                self._index_in(op.t_out, tensor_tracker, stride_tracker, all_dims, unro_tracker, \
                                const_tracker, offset_tracker[op], op_out_t_inds[op])

                self._finish_reduce(op_out_t_inds[op], dim_deps_out[op], dim_deps_in[op], all_dims, min_red_ax)

                op_out_t_inds[op].clear()
            else:
                self._index_in(op.t_out, tensor_tracker, stride_tracker, all_dims, unro_tracker, \
                                const_tracker, offset_tracker[op], op_out_t_inds[op])
                
                self._finish_una_bin(op_out_t_inds[op], op_link[op])


        for c in const_tracker.values():
            ast.append(c)

        for g in global_map:
            ast.append(g)

        for l in local_map:
            ast.append(l)

        for dim in ordered_dims:
            for op in queue:
                if dim in dim_deps_in[op]:
                    for dep in dim_deps_in[op][dim]:
                        ast.append(dep)

            if dim in unro_tracker:
                continue
            if (dim.op is MuOp.G_D) or (dim.op is MuOp.L_D):
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
        
        for dim in reversed(ordered_dims):
            if (dim.deps[1].value != "1") and (dim not in unro_tracker) and (dim.op is not MuOp.G_D) and (dim.op is not MuOp.L_D):
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


    def _val_opts(self, opt: Opt, red_axes: list[int], global_shape: list[int]) -> None:
        tesselate = opt.tesselate
        unroll = opt.unroll
        permute = opt.permute
        g_dim = opt.globals
        l_dim = opt.locals

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
            if (d in g_dim) or (d in l_dim):
                raise ValueError("You can't unroll a global or local dimension.")
            else:
                dim_set.add(d)

        red_set = set()

        if len(red_axes) > 0:
            offset = 0
            tes_set = sorted(set([i[0] for i in tesselate]))
            for i in tes_set:
                if i in red_axes:
                    red_set.add(i + offset)
                    red_set.add(i + offset + 1)
                offset += 1

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
                if len(red_set) > 0:
                    if red_set != set(permute[-len(red_set):len(permute)]):
                        raise ValueError("The reduce dimensions aren't the most nested (last) in the permutation given.")
                elif set(red_axes) != set(permute[-len(red_axes):len(permute)]):
                    raise ValueError("The reduce dimensions aren't the most nested (last) in the permutation given.")

        dim_set.clear()

        if len(g_dim) > 0:
            if len(l_dim) != len(g_dim):
                    raise ValueError("You must have the same number of global and local dimensions.")
            for dim in g_dim:
                if (dim < 0) or (dim > num_dims):
                    raise ValueError("You can't have a global dimension that is outside the provided number of dimensions.")
                if dim in red_set:
                    raise ValueError("A reduce dimension can't be a global dimension.")
                if dim in dim_set:
                    raise ValueError("You can't use the same dimension twice as a global dimension.")
                dim_set.add(dim)

        dim_set.clear()

        if len(l_dim) > 0:
            if len(g_dim) == 0:
                raise ValueError("You must have global dimensions in order to have local dimensions.")
            for dim in l_dim:
                if (dim < 0) or (dim > num_dims):
                    raise ValueError("You can't have a local dimension that is outside the provided number of dimensions.")
                if dim in red_set:
                    raise ValueError("A reduce dimension can't be a local dimension.")
                if dim in dim_set:
                    raise ValueError("You can't use the same dimension twice as a local dimension.")
                dim_set.add(dim)
            if len(dim_set.intersection(g_dim)) > 0:
                raise ValueError("The global and local dimensions cannot share axes.")


    def _define_ctx_and_args(self, queue: list[Op], ast: list[IR], ctx: dict[str, list[Tensor]], tensor_tracker: dict[Tensor, IR]) -> None:
        tensor_count: int = 0

        for op in queue:
            for parent in op.t_in:
                if parent in ctx:
                    tensor_tracker[parent].op = MuOp.T_ARG
                    tensor_tracker[parent].value = f"temp_in_{tensor_count}"
                    tensor_tracker[parent].dtype = parent.dtype
                    ast.pop(ast.index(tensor_tracker[parent]))
                    ctx[parent] = "T"
                else:
                    tensor_tracker[parent] = IR(MuOp.ARG, parent.dtype + "*", f"in_{tensor_count}", [])
                    ast.append(tensor_tracker[parent])
                    ctx[parent] = "L"
                tensor_count += 1
            
            tensor_tracker[op.t_out] = IR(MuOp.ARG, op.t_out.dtype + "*", "out", [])
            ast.append(tensor_tracker[op.t_out])
            ctx[op.t_out] = "S"


    def _check_consts(self, val: str, const_tracker: dict[str, IR]) -> None:
        if val not in const_tracker:
            const_tracker[val] = IR(MuOp.CONST, "int", val, [])


    def _update_and_map_dims(
                                self, 
                                global_shape: list[int], 
                                red_axes: list[int], 
                                const_tracker: dict[str, IR], 
                                ctx: dict[Tensor, str], 
                                opt: Opt
                            ) -> tuple[list[IR], list[IR], dict[Tensor, list[int]], dict[IR, int], list[int], list[int]]:
        tesselate = opt.tesselate
        unroll = opt.unroll
        permute = opt.permute

        tess_dims: list[int] = [i[0] for i in tesselate]
        tess_factors: list[int] = [i[1] for i in tesselate]

        new_global_shape: list[int] = []
        new_red_axes: list[int] = []
        all_dims: list[IR] = []

        ind_count: int = 0

        self._check_consts("0", const_tracker)

        for ind, dim in enumerate(global_shape):
            if ind in tess_dims:
                if ind in red_axes:
                    new_red_axes.append([ind_count, ind_count + 1])
                    muop_1: MuOp = MuOp.N_R
                    muop_2: MuOp = MuOp.N_R
                    axis_name_1: str = f"r_axis_{ind_count}"
                    axis_name_2: str = f"r_axis_{ind_count + 1}"
                else:
                    muop_1: MuOp = MuOp.N_D
                    muop_2: MuOp = MuOp.N_D
                    axis_name_1: str = f"axis_{ind_count}"
                    axis_name_2: str = f"axis_{ind_count + 1}"

                new_val: int = dim // tess_factors[tess_dims.index(ind)]
                t_factor: int = tess_factors[tess_dims.index(ind)]

                new_global_shape.append(new_val)
                new_global_shape.append(t_factor)

                self._check_consts(str(new_val), const_tracker)
                self._check_consts(str(t_factor), const_tracker)

                all_dims.append(IR(muop_1, "", axis_name_1, [const_tracker["0"], const_tracker[str(new_val)]]))
                all_dims.append(IR(muop_2, "", axis_name_2, [const_tracker["0"], const_tracker[str(t_factor)]]))

                ind_count += 2
            else:
                if ind in red_axes:
                    new_red_axes.append([ind_count])
                    muop: MuOp = MuOp.N_R
                    axis_name: str = f"r_axis_{ind_count}"
                else:
                    muop = MuOp.N_D
                    axis_name = f"axis_{ind_count}"

                self._check_consts(str(dim), const_tracker)

                new_global_shape.append(dim)

                all_dims.append(IR(muop, "", axis_name, [const_tracker["0"], const_tracker[str(dim)]]))

                ind_count += 1

        assert functools.reduce(operator.mul, global_shape) == functools.reduce(operator.mul, new_global_shape), "The reshaped volume does not match that of the original shape."


        unro_tracker: dict[IR, int] = {}

        for dim in unroll:
            unro_tracker[all_dims[dim]] = int(all_dims[dim].deps[1].value)


        ordered_dims: list[IR] = []
        flat_r_a = [dim for set_d in new_red_axes for dim in set_d]

        if len(permute) > 0:
            temp_order = [all_dims[i] for i in permute]
            ordered_dims = temp_order
        else:
            length_set = set([i for i in range(len(all_dims))])
            non_red_axes = length_set.symmetric_difference(flat_r_a)
            all_axes = list(non_red_axes) + flat_r_a

            temp_order = [all_dims[i] for i in all_axes]
            ordered_dims = temp_order


        stride_tracker: dict[Tensor, list[int]] = {}

        for ten in ctx:
            shape = [i for i in ten.view]
            stride = [i for i in ten.stride]
            for red_dim in red_axes:
                if len(stride) < len(global_shape):
                    shape.insert(red_dim, global_shape[red_dim])
                    stride.insert(red_dim, 0)
            
            new_shape, new_stride = self._merge_adjacent_contiguous_strides(shape, stride)

            stride_tracker[ten] = self._create_desired_shape_strides(new_global_shape, new_shape, new_stride)

        return (all_dims, ordered_dims, stride_tracker, unro_tracker, new_global_shape, new_red_axes)


    def _merge_adjacent_contiguous_strides(self, shape: list[int], stride: list[int]) -> tuple[list[int], list[int]]:
        new_shape = []
        new_stride = []

        temp_sh = None

        for sh, st in zip(reversed(shape), reversed(stride)):
            if st == 0:
                if len(new_stride) > 0:
                    if new_stride[-1] == 0:
                        temp_sh *= sh
                        new_shape[-1] = temp_sh
                        continue
                new_shape.append(sh)
                new_stride.append(st)
                temp_sh = sh
            else:
                if len(new_stride) > 0:
                    if new_stride[-1] != 0:
                        if st == (new_stride[-1] * temp_sh):
                            temp_sh *= sh
                            new_shape[-1] = temp_sh
                            continue
                new_shape.append(sh)
                new_stride.append(st)
                temp_sh = sh

        if 1 in new_shape:
            raise ValueError("A '1' in the tensor's shape could not be merged with its surrounding dimensions.")

        new_shape.reverse()
        new_stride.reverse()

        return (new_shape, new_stride)


    def _create_desired_shape_strides(self, desired_shape: list[int], new_shape: list[int], new_stride: list[int]) -> list[int]:
        assert functools.reduce(operator.mul, desired_shape) == functools.reduce(operator.mul, new_shape), "The reshaped volume does not match the original."

        created_shape = []
        created_stride = []

        for i in reversed(desired_shape):
            if new_shape[-1] // i == 0:
                raise ValueError("The contiguous section being reshaped is too small to accommodate the new dimension.")
            elif new_shape[-1] // i > 1:
                created_shape.append(i)
                new_shape[-1] = new_shape[-1] // i
            else:
                created_shape.append(i)
                new_shape.pop()

            if len(created_stride) == 0:
                created_stride.append(new_stride[-1])
            elif new_stride[-1] == 0:
                created_stride.append(0)
            elif (created_stride[-1] == 0) and (new_stride[-1] != 0):
                created_stride.append(new_stride[-1])
            else:
                created_stride.append(created_shape[-2] * created_stride[-1])

            if len(new_shape) < len(new_stride):
                new_stride.pop()

        created_stride.reverse()

        return created_stride


    def _map_parallel_dims(self, globals: list[int], locals: list[int], all_dims: list[IR], max_thread_limit: int, const_tracker: dict[str, IR]) -> tuple[list[IR], list[IR]]:
        if len(globals) == 0:
            return ([], [])

        global_dim_bounds: list[int] = []
        local_dim_bounds: list[int] = []

        for g, l in zip(globals, locals):
            global_dim_bounds.append(int(all_dims[g].deps[1].value))
            local_dim_bounds.append(int(all_dims[l].deps[1].value))

        self.blocks_per_grid = functools.reduce(operator.mul, global_dim_bounds)
        self.threads_per_block = functools.reduce(operator.mul, local_dim_bounds)

        assert self.threads_per_block <= max_thread_limit, "A block can't have more threads than the hardware limit."

        global_setup: list[IR] = []

        global_setup.append(IR(MuOp.BUILT_IN, "int", "blockIdx.x", []))

        for num, g in enumerate(globals):
            all_dims[g].op = MuOp.G_D
            all_dims[g].value = f"g_dim_{num}"

            inclusive_prod: int = functools.reduce(operator.mul, global_dim_bounds[num:])
            if len(global_dim_bounds[num + 1:]) > 0:
                exclusive_prod: int = functools.reduce(operator.mul, global_dim_bounds[num + 1:])
            else:
                exclusive_prod = 1

            self._check_consts(str(inclusive_prod), const_tracker)
            self._check_consts(str(exclusive_prod), const_tracker)

            global_setup.append(IR(MuOp.MOD, "int", "", deps=[global_setup[0], const_tracker[str(inclusive_prod)]]))
            global_setup.append(IR(MuOp.FL_DIV, "int", "", deps=[global_setup[-1], const_tracker[str(exclusive_prod)]]))
            global_setup.append(IR(MuOp.STORE, "int", all_dims[g].value, deps=[all_dims[g], global_setup[-1]]))
        

        local_setup: list[IR] = []

        local_setup.append(IR(MuOp.BUILT_IN, "int", "threadIdx.x", []))

        for num, l in enumerate(locals):
            all_dims[l].op = MuOp.L_D
            all_dims[l].value = f"l_dim_{num}"

            inclusive_prod: int = functools.reduce(operator.mul, local_dim_bounds[num:])
            if len(local_dim_bounds[num + 1:]) > 0:
                exclusive_prod: int = functools.reduce(operator.mul, local_dim_bounds[num + 1:])
            else:
                exclusive_prod = 1

            self._check_consts(str(inclusive_prod), const_tracker)
            self._check_consts(str(exclusive_prod), const_tracker)

            local_setup.append(IR(MuOp.MOD, "int", "", deps=[local_setup[0], const_tracker[str(inclusive_prod)]]))
            local_setup.append(IR(MuOp.FL_DIV, "int", "", deps=[local_setup[-1], const_tracker[str(exclusive_prod)]]))
            local_setup.append(IR(MuOp.STORE, "int", all_dims[l].value, deps=[all_dims[l], local_setup[-1]]))

        return (global_setup, local_setup)


    def _index_in(
                    self, 
                    tensor: Tensor, 
                    tensor_tracker: dict[Tensor, IR], 
                    stride_tracker: dict[Tensor, list[int]], 
                    all_dims: list[IR], 
                    unro_tracker: dict[IR, int], 
                    const_tracker: dict[str, IR], 
                    offset_tracker: list[int], 
                    op_t_inds: list[list[IR]], 
                ) -> None:

        stride = stride_tracker[tensor]
        opt_dim_vals = []
        temp_offset = 1

        for dim in all_dims:
            if dim in unro_tracker:
                opt_dim_vals.append(range(unro_tracker[dim]))
                temp_offset *= unro_tracker[dim]
            else:
                opt_dim_vals.append([dim])
        
        temp_count: int = 0
        offset_tracker.append(temp_offset)

        for mapping in product(*opt_dim_vals):
            if tensor_tracker[tensor].op is MuOp.T_ARG:
                temp_t: IR = tensor_tracker[tensor]
                op_t_inds.append([IR(MuOp.T_ARG, temp_t.dtype, temp_t.value + f"_{temp_count}", [])])
                temp_count += 1
                continue

            temp_index_list = []
            previous_term = None

            for dim, st in zip(mapping, stride):
                if isinstance(dim, IR):
                    if st == 0:
                        continue
                    self._check_consts(str(st), const_tracker)
                    temp_op: IR = IR(MuOp.MUL, "int", "", [dim, const_tracker[str(st)]])
                    temp_index_list.append(temp_op)
                else:
                    combined_stride = dim * st
                    self._check_consts(str(combined_stride), const_tracker)
                    temp_op = const_tracker[str(combined_stride)]

                if previous_term is not None:
                    temp_op = IR(MuOp.ADD, "int", "", [previous_term, temp_op])
                    temp_index_list.append(temp_op)
                
                previous_term = temp_op

            temp_index_list.append(IR(MuOp.LOAD, tensor.dtype, tensor_tracker[tensor].value, [tensor_tracker[tensor], previous_term]))

            op_t_inds.append(temp_index_list)


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
                        red_ax: int, 
                        red_map_factor: int, 
                        all_dims: list[IR], 
                        const_tracker: dict[str, IR], 
                    ) -> None:
        match_init_val: dict[MuOp, str] = {MuOp.ADD: "0", MuOp.MAX: "-INFINITY", MuOp.MIN: "INFINITY"}
        init_val = match_init_val[op]
        self._check_consts(init_val, const_tracker)

        last_ele_ind = len(op_in_t_inds[0]) - 1
        data_type = op_in_t_inds[0][last_ele_ind].dtype

        red_ax_ir: IR = all_dims[red_ax]

        dim_deps_in[red_ax_ir] = []

        num_temp: int = int(len(op_in_t_inds) / red_map_factor)

        for i in range(num_temp):
            dim_deps_in[red_ax_ir].append(IR(MuOp.TEMP, data_type, f"temp_{i}", [const_tracker[str(init_val)]]))

        for num, temp in enumerate(dim_deps_in[red_ax_ir]):
            for i in range(red_map_factor):
                temp_ir = IR(op, data_type, "", [temp, op_in_t_inds[num + (i * num_temp)][last_ele_ind]])
                op_link.append(temp_ir)
                op_link.append(IR(MuOp.STORE, data_type, temp.value, [temp, temp_ir]))


    def _finish_reduce(
                        self, 
                        op_out_t_inds: list[list[IR]], 
                        dim_deps_out: list[list[IR]], 
                        dim_deps_in: list[list[IR]], 
                        all_dims: list[IR], 
                        red_ax: int
                    ) -> None:
        last_ele_ind = len(op_out_t_inds[0]) - 1
        data_type = op_out_t_inds[0][last_ele_ind].dtype

        red_ax_ir: IR = all_dims[red_ax]

        dim_deps_out[red_ax_ir] = []

        for out_t, in_dep in zip(op_out_t_inds, dim_deps_in[red_ax_ir]):
            for ir in out_t:
                dim_deps_out[red_ax_ir].append(ir)

            dim_deps_out[red_ax_ir].append(IR(MuOp.STORE, data_type, out_t[last_ele_ind].value, [out_t[last_ele_ind], in_dep]))


    def _finish_una_bin(self, op_out_t_inds: list[list[IR]], op_link: list[IR]) -> None:
        last_ele_ind = len(op_out_t_inds[0]) - 1
        data_type = op_out_t_inds[0][last_ele_ind].dtype

        for num, t_out in enumerate(op_out_t_inds):
            t_out.append(IR(MuOp.STORE, data_type, t_out[last_ele_ind].value, [t_out[last_ele_ind], op_link[num]]))

