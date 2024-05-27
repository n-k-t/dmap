from __future__ import annotations
import functools
import operator
from typing import Optional

from device import Device
from memory import Memory
from static import DType


# A subset of tensor that is lazy and helps abstracts away the internals.
class LazyTensor():
    def __init__(
            self, 
            shape: list[int], 
            dtype: DType, 
            device: Device, 
            memory: Optional[Memory] = None
        ) -> LazyTensor:
        self.dtype = dtype
        self.device = device

        # Ensure that the shape is valid.
        self.validate_shape(shape)

        self.shape = shape

        # Calculate the stride from the provided shape.
        self.stride: list[int] = self.stride_from_shape(shape)

        # Populate a hidden flag indicating whether or not the stored shape and stride are contiguous.
        self._contiguous: bool = self.check_contiguous(self.stride, shape)

        # Give the option to utilize a pre-initialized memory object, otherwise create a new one.
        if memory:
            self.memory = memory
        else:
            self.memory: Memory = Memory(functools.reduce(operator.mul, shape), dtype, device)

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

    # A function that determines whether or not the lazy tensor is contiguous.
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