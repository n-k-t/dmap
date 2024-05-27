from __future__ import annotations
from typing import List, Optional, Union
import functools
import operator

from device import Device
from lazy import LazyTensor
from identifiers import DType


# The class from which all operation classes inherit their initialization and application methods.
class Op():
    def __init__(
            self, 
            *tensors: Tensor
        ) -> Op:
        self.parents = tensors

    # A method available to all child classes that applies the operation class to the input tensors and produces the resulting tensor.
    @classmethod
    def apply(
            self, 
            *tensors: Tensor, 
            **kwargs: Optional[List[int]]
        ) -> Tensor:
        # Ensure that all tensors being operated upon are located on the same device.
        op_device: str = tensors[0].device.standard()
        if len(tensors) > 1:
            assert all(t.device.standard() == op_device for t in tensors), \
                "The tensors provided for the operation must all be on the same device."

        # Instantiate the operation and set the operand tensors.
        op_node: Op = self(*tensors)

        # Get the resulting lazy tensor from the forward pass.
        res_lazy_t: LazyTensor = op_node.forward(*[t.tdata for t in tensors], **kwargs)

        # Create the resulting tensor and add its source operation (for the backward pass).
        new_tensor: Tensor = Tensor(res_lazy_t, res_lazy_t.dtype, op_device)
        new_tensor._src_op = op_node

        return new_tensor


# Order the importing to resolve a circular import.
import op


# The base tensor class (the highest level of abstraction).
class Tensor():
    def __init__(
            self, 
            tdata: Union[LazyTensor, List[int]], 
            dtype: DType = DType.float32, 
            device: Optional[str] = None
        ) -> Tensor:
        self.dtype = dtype

        # Associate the tensor with the base device if none is specified.
        if device is None:
            self.device = Device()
        else:
            self.device = Device(device)

        # If the tensor data is not a lazy tensor instance, then convert it into one.
        if isinstance(tdata, LazyTensor):
            self.tdata = tdata
        elif isinstance(tdata, List):
            # Verify that the specified tensor shape includes a greater number of dimensions than zero.
            assert len(tdata) > 0, \
                "A tensor cannot have no dimensionality."

            self.tdata = LazyTensor(tdata, dtype, self.device)

        # Create a hidden "source operation" field that can be filled (for tracing) if an operation is applied.
        self._src_op: Optional[Op] = None

    # ------- Utility Functions ------- #

    # Verify that the two tensors being operated upon are the same shape.
    def _shape_check(
            self, 
            t2: Tensor
        ) -> None:
        if self.tdata.shape != t2.tdata.shape:
            raise ValueError("The two tensors being operated upon do not have identical shapes.")

    # ------- Binary Ops ------- #

    # Apply the addition operation to the current object and one other tensor.
    def add(
            self, 
            t2: Tensor
        ) -> Tensor:
        self._shape_check(t2)
        return op.Add.apply(self, t2)

    # Apply the division operation to the current object and one other tensor.
    def div(
            self, 
            t2: Tensor
        ) -> Tensor:
        self._shape_check(t2)
        return op.Div.apply(self, t2)

    # Apply the multiplication operation to the current object and one other tensor.
    def mul(
            self, 
            t2: Tensor
        ) -> Tensor:
        self._shape_check(t2)
        return op.Mul.apply(self, t2)

    # Apply the subtraction operation to the current object and one other tensor.
    def sub(
            self, 
            t2: Tensor
        ) -> Tensor:
        self._shape_check(t2)
        return op.Sub.apply(self, t2)

    # ------- Memory Alteration Ops ------- #

    # Apply a safe reshape to the tensor meaning contiguity and size of the memory region must be maintained.
    #### NOTE: There is no copy involved here.
    def safe_reshape(
            self, 
            new_shape: List[int]
        ) -> Tensor:
        # Verify that the starting tensor is contiguous, otherwise a safe reshape cannot occur.
        assert self.tdata._contiguous, \
            "The current shape is not contiguous, therefore it can't safely be reshaped."

        # Calculate the number of elements in the provided tensor shape and verify that the new size of the memory region matches the old.
        #### NOTE: If the conditions in this function and the lazy tensor init are satisfied, then the output must be contiguous.
        new_ele_cnt: int = functools.reduce(operator.mul, new_shape)
        assert new_ele_cnt == functools.reduce(operator.mul, self.tdata.shape), \
            "The specified reshape cannot be performed as it results in a different sized region of memory than the original allocation."

        return op.SafeReshape.apply(self, new_shape = new_shape)

    # Apply an unsafe reshape to the tensor meaning there are no restrictions to alterations made to the memory region (outside of those required to be a tensor).
    #### NOTE: There is no copy involved here.
    def unsafe_reshape(
            self, 
            new_shape: List[int], 
            new_stride: List[int]
        ) -> Tensor:
        # Verify that the provided shape and stride have an equal number of dimensions.
        assert len(new_shape) == len(new_stride), \
            "The specified reshape cannot be performed because the number of dimensions provided for the shape and stride do not match."

        return op.UnafeReshape.apply(self, new_shape = new_shape, new_stride = new_stride)

    # ------- Unary Ops ------- #

    # Apply the exponentiation operation to the current object (base tensor, exponent e).
    def exp(
            self
        ) -> Tensor:
        return op.Exp.apply(self)

    # Apply the logarithm operation to the current object (base e).
    def log(
            self
        ) -> Tensor:
        return op.Log.apply(self)