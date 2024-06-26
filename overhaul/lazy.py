from __future__ import annotations
import functools
import operator
from typing import List, Optional, Union

from device import Device
from memory import Memory
from identifiers import Binary, DType, MemoryAlter, MemoryMove, Reduce, Unary


# A subset of tensor that is lazy and helps abstracts away the internals.
class LazyTensor():
    def __init__(
            self, 
            shape: list[int], 
            dtype: DType, 
            device: Device, 
            stride: Optional[List[int]] = None, 
            parents: List[LazyTensor] = [], 
            src_op: Optional[Union[Binary, MemoryAlter, MemoryMove, Reduce, Unary]] = None,
            memory: Optional[Memory] = None, 
            extra: Optional[int] = None
        ) -> LazyTensor:
        self.dtype = dtype
        self.device = device
        self.parents = parents

        # Store extra information (i.e. reduce axis).
        self.extra = extra

        # A flag indicating whether or not the LazyTensor has been evaluated or not (defaults to false).
        self.evaluated: bool = False

        # A flag indicating whether or not the LazyTensor was forced to evaluate during the scheduling process (defaults to False).
        self.force_evaluated: bool = False

        # Set the source operation.
        self.src_op = src_op

        # If no source operation is provided, then the tensor is a result of a load operation (and evaluated becomes a schedule barrier).
        if src_op is None:
            self.src_op = MemoryMove.LOAD
            self.evaluated = True
        elif isinstance(src_op, MemoryMove):
            self.evaluated = True
        else:
            self.src_op = src_op

        # Ensure that the provided shape is valid.
        self.validate_shape(shape)

        self.shape = shape

        # Calculate the stride from the provided shape if none is provided, otherwise use the given one.
        if stride is None:
            self.stride = self.stride_from_shape(shape)
        else:
            self.stride = stride

        # Populate a hidden flag indicating whether or not the stored shape and stride are contiguous.
        self._contiguous: bool = self.check_contiguous(self.stride, shape)

        # Check for a pre-initialized memory object to use, otherwise create a new one.
        if memory is None:
            self.memory: Memory = Memory(functools.reduce(operator.mul, shape), shape, self.stride, dtype, device)
        else:
            self.memory = memory

    # A method that verifies that the shape are all positive integers greater than 0. 
    def validate_shape(
            self, 
            shape: list[int]
        ) -> None:
        if not all(isinstance(x, int) for x in shape):
            raise TypeError("Your memory shape descriptors are not all integers.")
        if functools.reduce(operator.mul, shape) <= 0:
            raise ValueError("You can't have negatives or zeros describing your memory structure.")

    # Calculates the stride from the provided shape.
    def stride_from_shape(
            self, 
            shape: list[int]
        ) -> list[int]:
        stride = [1]
        for i in range(len(shape) - 1):
            stride.append(stride[i] * shape[-(i + 1)])
        stride.reverse()
        return stride

    # A function that determines whether or not the LazyTensor is contiguous.
    def check_contiguous(
            self, 
            stride: list[int], 
            shape: list[int]
        ) -> bool:
        if not stride[-1] == 1:
            return False
        if not all((stride[i - 1] == (stride[i] * shape[i])) for i in range(len(stride) - 1, 0, -1)):
            return False
        return True

    # Cast a LazyTensor to a new data type.
    def cast(
            self, 
            new_dtype: DType
        ) -> LazyTensor:
        return LazyTensor(self.shape, new_dtype, self.device, parents = [self], src_op = MemoryAlter.CAST)

    # Create a contiguous copy of a LazyTensor on the same device.
    def contiguous(
            self
        ) -> LazyTensor:
        return LazyTensor(self.shape, self.dtype, self.device, parents = [self], src_op = MemoryMove.CONTIGUOUS)

    # Create a copy of a LazyTensor on a new device.
    def copy(
            self, 
            destination: Device
        ) -> LazyTensor:
        return LazyTensor(self.shape, self.dtype, destination, self.stride, [self], MemoryMove.COPY)