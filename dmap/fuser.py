from dmap.tensor import Op, Tensor
from dmap.ops import Binary, Fusion, Memory, Movement, Reduce, Unary

class Fuser:
    def __init__(self, head: Tensor, fuse: bool = False) -> None:
        self.tokens = self._traverse(head, fuse)
        self.names = []
        for token in self.tokens:
            self._gen_kernel_name(token)


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
            return self._fuse(self._apply_restrictions(top_sort(tensor, set(), [])))
        else:
            return self._apply_restrictions(top_sort(tensor, set(), []))


    def _fuse(self, ops: list[Op]) -> list[Op]:
        index: int = 0

        while index < (len(ops) - 1):
            if isinstance(ops[index].op, Binary) and isinstance(ops[index + 1].op, Reduce) and (ops[index].t_out in ops[index + 1].t_in):
                op_holder = Op(Fusion.ELE_RED)
                op_holder.num_flop = ops[index].num_flop + ops[index + 1].num_flop
                op_holder.fus_ops = ops[index: index + 2]
                op_holder.axis += ops[index + 1].axis
                ops.pop(index)
                ops[index] = op_holder
            index += 1

        return ops


    def _apply_restrictions(self, token_stream: list[Op]) -> list[Op]:        
        return [token for token in token_stream if (not isinstance(token.op, Memory)) and (not isinstance(token.op, Movement))]
    

    def _gen_kernel_name(self, op: Op) -> None:
        if isinstance(op.op, Unary) or isinstance(op.op, Binary):
            temp_name = "_".join(["E"] + [str(dim) for dim in op.t_in[0].view] + ["to"] + [str(dim) for dim in op.t_out.view] + ["v"])
        elif isinstance(op.op, Reduce):
            temp_name = "_".join(["R"] + [str(dim) for dim in op.t_in[0].view] + ["to"] + [str(dim) for dim in op.t_out.view] + ["v"])
        elif isinstance(op.op, Fusion):
            temp_name = "_".join(["F"] + [str(dim) for dim in op.fus_ops[0].t_in[0].view] + ["to"] + [str(dim) for dim in op.fus_ops[-1].t_out.view] + ["v"])
        
        repetitions: int = 0
        for name in self.names:
            if temp_name in name:
                repetitions += 1
        
        self.names.append(temp_name + str(repetitions))