from __future__ import annotations
import functools
import operator
from copy import deepcopy
from dmap.memory import Memory
from dmap.ops import Op, Memory, Movement, Unary, Binary, Reduce
from typing import Optional


class Tensor:
    def __init__(
            self, 
            view: list[int], 
            stride: list[int] = [], 
            offset: int = 0,
            parents: list[Tensor] = [],
            op: Optional[Op] = None
        ) -> None:
        # Control flow for setting both the operation and memory fields.
        self.validate_shape(view)
        if not op:
            self.op = Op(Memory.LOAD) # Need to make this an actual memory object
            self.stride = self.stride_from_view(view)
            # self._memory: Memory = Memory(shape, True) # Remove memory at the end of the day.
        elif op.op != Movement.RESHAPE_U:
            self.op = op
            self.stride = self.stride_from_view(view)
            # self._memory = Memory(shape, True)
        else:
            self.op = op
            self.stride = stride
            # self._memory = Memory(shape, False, stride)
        self.view = view
        self.offset = offset
        self.mask: dict[str, list[list[int | str]]] = {"Axes": [], "Indices": []} # list -> [axis, "gt/lt/eqt", index]
        self.contiguous: bool = self.check_contiguous(self.stride, view)
        self.dtype: str = "float"
        self.parents = parents
        self.children: list[Tensor] = []

    # Initialization Functions
    def validate_shape(self, shape: list[int]) -> None:
        if any(isinstance(x, bool) for x in shape):
            raise TypeError("Your memory shape descriptors are not all integers.")
        if not all(isinstance(x, int) for x in shape):
            raise TypeError("Your memory shape descriptors are not all integers.")
        if functools.reduce(operator.mul, shape) <= 0:
            raise ValueError("You can't have negatives or zeros describing your memory structure.")


    def check_contiguous(self, stride: list[int], view: list[int]) -> bool:
        if not stride[-1] == 1:
            return False
        if not all((stride[i - 1] == (stride[i] * view[i])) for i in range(len(stride) - 1, 0, -1)):
            return False
        return True


    def stride_from_view(self, view: list[int]) -> list[int]:
        stride = [1]
        for i in range(len(view) - 1):
            stride.append(stride[i] * view[-(i + 1)])
        stride.reverse()
        return stride


    # Memory Operations
    def safe_reshape(self, new_shape: list[int]) -> Tensor:
        new_size = functools.reduce(operator.mul, new_shape)
        assert new_size > 0, "The specified reshape cannot be performed as it has atleast one dimension of size zero."
        assert new_size == functools.reduce(operator.mul, self.view), "The specified reshape cannot be performed as it results in a different sized region of memory than the original allocation."
        # operation = MovementOp("SAFE_RESHAPE", new_shape)
        operation = Op(Movement.RESHAPE_S, new_shape)
        # TODO: Can't remove these labels because stride is ahead. Maybe move it to the back.
        child = Tensor(view = new_shape, parents = [self], op = operation)
        self.children.append(child)
        return child


    # Permute and expand are unsafe reshapes.
    def _unsafe_reshape(self, new_shape: list[int], new_stride: list[int]) -> Tensor:
        assert functools.reduce(operator.mul, new_shape) > 0, "The specified reshape cannot be performed as it has atleast one dimension of size zero."
        assert len(new_shape) == len(new_stride), "The specified reshape cannot be performed because the number of dimensions provided for the shape and stride do not match."
        # operation = MovementOp("UNSAFE_RESHAPE", new_shape, new_stride)
        operation = Op(Movement.RESHAPE_U, new_shape, new_stride)
        child = Tensor(view = new_shape, stride = new_stride, parents = [self], op = operation)
        self.children.append(child)
        return child


    # Binary Operations
    def _binary_operation(self, tensor_2: Tensor, op_type: Binary) -> Tensor:
        assert self.view == tensor_2.view, "The operation cannot be performed because the two tensors do not have identical shapes."
        # operation = BinaryOp(op_type)
        num_flop: int = functools.reduce(operator.mul, self.view)
        operation = Op(op_type, flop = num_flop)
        # TODO: Can't remove these labels because stride is ahead. Maybe move it to the back.
        child = Tensor(view = self.view, parents = [self, tensor_2], op = operation)
        self.children.append(child)
        tensor_2.children.append(child)
        return child
    

    def add(self, tensor_2: Tensor) -> Tensor:
        return self._binary_operation(tensor_2, Binary.ADD)
    

    def div(self, tensor_2: Tensor) -> Tensor:
        return self._binary_operation(tensor_2, Binary.DIV)
    

    def mul(self, tensor_2: Tensor) -> Tensor:
        return self._binary_operation(tensor_2, Binary.MUL)
    

    def sub(self, tensor_2: Tensor) -> Tensor:
        return self._binary_operation(tensor_2, Binary.SUB)
    

    # Reduction Operations
    def _reduction(self, axis: int, op_type: Reduce) -> Tensor:
        assert axis >= 0, "The reduction operation cannot be perfomed along a negative axis."
        assert axis <= len(self.view) - 1, "The reduction operation cannot be performed because the axis provided is greater than the number of dimensions in the tensor."
        # operation = ReduceOp(op_type, axis)
        if self.view[axis] > 1:
            op_adjustment: list[int] = deepcopy(self.view)
            op_adjustment[axis] -= 1
            num_flop: int = functools.reduce(operator.mul, op_adjustment)
        else:
            num_flop: int = functools.reduce(operator.mul, self.view)
        operation = Op(op_type, axis = axis, flop = num_flop)
        if len(self.view) == 1:
            new_shape = [1]
        else:
            new_shape = deepcopy(self.view)
            new_shape.pop(axis)
        child = Tensor(view = new_shape, parents = [self], op = operation)
        self.children.append(child)
        return child
    

    def max(self, axis: int) -> Tensor:
        return self._reduction(axis = axis, op_type = Reduce.MAX)
    

    def min(self, axis: int) -> Tensor:
        return self._reduction(axis = axis, op_type = Reduce.MIN)


    def sum(self, axis: int) -> Tensor:
        return self._reduction(axis = axis, op_type = Reduce.SUM)
    
    
    # Topological Sort
    def _topological_sort(self) -> list[Tensor]:
        def top_sort_util(tensor, visited, stack) -> list[Tensor]:
            visited.add(tensor)
            for parent in tensor.parents:
                if parent not in visited:
                    top_sort_util(parent, visited, stack)
            stack.append(tensor)
            return stack
        return top_sort_util(self, set(), [])