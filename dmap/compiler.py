from copy import deepcopy
import functools
import operator
from dmap.tensor import Tensor
from dmap.ops import Op, Reduce, Binary


class Compiler():
    def __init__(self, head: Tensor) -> None:
        self.tokens = self._topological_sort(head) # If the item is returned from a function, then the user
        # should know what the item is by looking at the return type of the function. This means that I should 
        # not need to declare the type here.

        # Reduce this and make sure it matches the tokens above.
        # It currently does not run for every single operation type.
        # I should record 0 for free operations.
        self.num_flop: list[int] = [self.calc_flop(token[1]) for token in self.tokens if isinstance(token[0].op, Reduce) or isinstance(token[0].op, Binary)]

    def calc_flop(self, tensor: Tensor) -> int:
        if isinstance(tensor.op.op, Reduce) and (tensor.parents[0].view[tensor.op.axis] > 1):
            op_adjustment: list[int] = deepcopy(tensor.parents[0].view)
            op_adjustment[tensor.op.axis] -= 1
            return functools.reduce(operator.mul, op_adjustment)
        else:
            return functools.reduce(operator.mul, tensor.parents[0].view)

    def _topological_sort(self, tensor: Tensor) -> list[tuple[Op, Tensor, list[Tensor]]]:
        def top_sort_util(tensor, visited, stack) -> list[Tensor]:
            visited.add(tensor)
            for parent in tensor.parents:
                if parent not in visited:
                    top_sort_util(parent, visited, stack)
            if tensor.op not in visited:
                visited.add(tensor.op)
                stack.append((tensor.op, tensor, tensor.parents))
            return stack
        return top_sort_util(tensor, set(), [])