from __future__ import annotations
from typing import Optional

from device import Device
from static import DType


# A class that oversees the memory occupied by a tensor.
class Memory():
    def __init__(
            self, 
            num_ele: int, 
            dtype: DType, 
            device: Device
        ) -> Memory:
        self.num_ele = num_ele
        self.dtype = dtype
        self.device = device

        # A flag that indicates whether or not the memory has been allocated (defaults to false).
        self.allocated: bool = False

        # A field that contains a pointer to the memory (not set until allocation).
        self.pointer: Optional[memoryview] = None