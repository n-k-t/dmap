from __future__ import annotations
from typing import List, Optional

from device import Device
from identifiers import DType


# A class that oversees the memory occupied by a tensor.
class Memory():
    def __init__(
            self, 
            num_ele: int, 
            true_shape: List[int], 
            true_stride: List[int], 
            dtype: DType, 
            device: Device
        ) -> Memory:
        self.num_ele = num_ele
        self.true_shape = true_shape
        self.true_stride = true_stride
        self.dtype = dtype
        self.device = device

        # A field that indicates the memory layout (defaults to row-major order like the C language).
        self.row_major: bool = True

        # A flag that indicates whether or not the memory has been allocated (defaults to false).
        self.allocated: bool = False

        # A field that contains a pointer to the memory (not set until allocation).
        self.pointer: Optional[memoryview] = None