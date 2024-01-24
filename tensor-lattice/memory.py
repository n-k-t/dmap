from __future__ import annotations
import functools
import operator

def validate_shape(
        shape: list[int]
    ) -> None:
    if any(isinstance(x, bool) for x in shape):
        raise TypeError("Your memory shape descriptors are not all integers.")
    if not all(isinstance(x, int) for x in shape):
        raise TypeError("Your memory shape descriptors are not all integers.")
    if functools.reduce(operator.mul, shape) <= 0:
        raise ValueError("You can't have negatives or zeros describing your memory structure.")
    
def check_contiguous(
        stride: list[int], 
        view: list[int]
    ) -> bool:
    if not stride[-1] == 1:
        return False
    if not all((stride[i - 1] == (stride[i] * view[i])) for i in range(len(stride) - 1, 0, -1)):
        return False
    return True

def stride_from_view(
        view: list[int]
    ) -> list[int]:
    stride = [1]
    for i in range(len(view) - 1):
        stride.append(stride[i] * view[-(i + 1)])
    stride.reverse()
    return stride

# Create a shape compare for binary ops and figure how to track for movement ops in relation to the original tensor
# Do we track indices to maintain order?
# I think no, instead we should just create a reshape and unshape reshape option (they can specify the stride here)
#### Maybe a permute as well? -- maybe unneeded, just include that in the unsafe?
# FIGURE OUT THE RESHAPING OF THESE TENSORS! - mapping essentially.
class Memory:
    def __init__(
            self, 
            shape: list[int], 
            safe_op: bool = True, 
            stride: list[int] | None = None
        ) -> Memory:
        validate_shape(shape = shape)
        # All operations are safe except for the UNSAFE_RESHAPE.
        #### Safety denotes that there is no change in stride of the same block of memory.
        #### It does not guarantee the same shape.
        if safe_op:
            self.view: list[int] = shape
            self.stride: list[int] = stride_from_view(view = self.view)
        else:
            self.view: list[int] = shape
            self.stride: list[int] = stride
        self._offset: int = 0
        self._mask: list[list[int]] | None = None
        self._contiguous: bool = check_contiguous(stride = self.stride, view = self.view)
        self._data_type: str = 'float'
        self._data_size: int = 4

