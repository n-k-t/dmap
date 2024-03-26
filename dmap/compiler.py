from dmap.tensor import Tensor

class Compiler():
    def __init__(self) -> None:
        pass

    # Topological Sort
    #TODO: does this work?
    def _topological_sort(self, tensor: Tensor) -> list[Tensor]:
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