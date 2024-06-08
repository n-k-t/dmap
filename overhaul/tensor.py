from __future__ import annotations
from typing import Dict, Generator, List, Optional, Union
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
            **kwargs: Optional[Union[bool, List[int]]]
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


# Order the inclusion of the op file to resolve a circular import.
import op


# The base tensor class (the highest level of abstraction).
class Tensor():
    def __init__(
            self, 
            tdata: Union[LazyTensor, List[int]], 
            dtype: DType = DType.float32, 
            device: Optional[str] = None, 
            req_grad: bool = False
        ) -> Tensor:
        self.dtype = dtype

        # Associate the tensor with the base device if none is specified.
        if device is None:
            self.device = Device()
        else:
            self.device = Device(device)

        # If the tensor data is not a lazy tensor instance, then convert it into one.
        if isinstance(tdata, LazyTensor):
            # Verify that the tensor is being created on the correct device.
            assert self.device.standard() == tdata.device.standard(), \
                f"The tensor is attempting to be made on the '{self.device.standard()}' from data located on the '{tdata.device.standard()}'."

            self.tdata = tdata
        elif isinstance(tdata, List):
            # Verify that the specified tensor shape includes a greater number of dimensions than zero.
            assert len(tdata) > 0, \
                "A tensor cannot have zero dimensions."

            self.tdata = LazyTensor(tdata, dtype, self.device)

        # If the gradient is required, then evaluate the LazyTensor.
        if req_grad:
            self.evaluate()

        self.req_grad = req_grad

        # Create a hidden variable that can be filled (for tracing) if an operation is applied to this instance of a tensor.
        self._src_op: Op = op.Instantiate()

        # Create a hidden variable that stores the gradient of the tensor with respect to some other tensor (used during backpropagation).
        self._grad: Optional[Tensor] = None


    # ------- Utility Functions ------- #

    # Verify that the two tensors being operted upon are located on the same device.
    def _device_check(
            self, 
            t2: Tensor
        ) -> None:
        if self.device.standard() != t2.device.standard():
            raise ValueError("The two tensors being operated upon are not located on the same device.")

    # Verify that the two tensors being operated upon are the same shape.
    def _shape_check(
            self, 
            t2: Tensor
        ) -> None:
        if self.tdata.shape != t2.tdata.shape:
            raise ValueError("The two tensors being operated upon do not have identical shapes.")

    # Verify that the two tensors being operated upon have the same data type.
    def _type_check(
            self, 
            t2: Tensor
        ) -> None:
        if self.dtype is not t2.dtype:
            raise ValueError("The two tensors being operated upon do not have identical data types.")

    # Verify that the reduction axis specified is legal (within the dimensional bounds of the tensor) and if so, then calculate the new shape.
    def _reduce_util(
            self, 
            axis: int, 
            keep_dim: bool
        ) -> None:
        # Make sure that the axis is non-negative.
        assert axis >= 0, \
            "The reduction operation cannot be perfomed along a negative axis."

        # Guarantee that the axis exists for the tensor.
        assert axis <= len(self.tdata.shape) - 1, \
            "The provided reduction axis index is greater than the number of dimensions in the tensor."

        # A tensor cannot collapse to zero dimensions.
        if len(self.tdata.shape) == 1:
            return [1]

        # Preserve the dimension if desired.
        if keep_dim is True:
            return [i if num != axis else 1 for num, i in enumerate(self.tdata.shape)]

        # Remove the dimension as expected.
        return [i for num, i in enumerate(self.tdata.shape) if num != axis]


    # ------- Abstraction Functions ------- #

    # Evaluate the lazy tensor.
    def evaluate(
            self
        ) -> None:
        self.tdata.evaluated = True


    # ------- Backpropagation Functions ------- #

    # Get a topological sort of all non-load operation resulting tensors in the DAG created by the operations specified below (calling ops in op.py).
    #### NOTE: A DAG is a directed acyclic graph and, in this case, represents a directed graph containing tensors and operations created by the user.
    def _top_sort(
            self
        ) -> List[Tensor]:
        # Create a utility function that recursively yields all tensors in the DAG with operand tensors (parents).
        def ts_util(
                tensor: Tensor, 
                t_set: set[Tensor]
            ) -> Generator[Tensor]:
            # Add the tensor called in the function to a set of visited tensors.
            t_set.add(tensor)

            # Check if the tensor has a source operation that is not its creation (if it does, then it will have parents for backpropagation).
            if not isinstance(tensor._src_op, op.Instantiate):
                # Iterate over all of the parent tensors of the result tensor.
                for parent in tensor._src_op.parents:
                    # If a parent is not already visited, then yield it to the list and call the generator on it.
                    if parent not in t_set:
                        yield from ts_util(parent, t_set)
                # Yield the tensor called in the function to the list.
                yield tensor

        # Gather all non-load operation tensors in an ordered, topological sort list (starting on the input tensor and creating an empty set).
        return list(ts_util(self, set()))


    # ------- Binary Ops ------- #

    # Apply the addition operation to the current object and one other tensor.
    def add(
            self, 
            t2: Tensor
        ) -> Tensor:
        # Validate the device, data type, and shapes.
        self._device_check(t2)
        self._type_check(t2)
        self._shape_check(t2)

        return op.Add.apply(self, t2)

    # Apply the division operation to the current object and one other tensor.
    def div(
            self, 
            t2: Tensor
        ) -> Tensor:
        # Validate the device, data type, and shapes.
        self._device_check(t2)
        self._type_check(t2)
        self._shape_check(t2)

        return op.Div.apply(self, t2)

    # Apply the multiplication operation to the current object and one other tensor.
    def mul(
            self, 
            t2: Tensor
        ) -> Tensor:
        # Validate the device, data type, and shapes.
        self._device_check(t2)
        self._type_check(t2)
        self._shape_check(t2)

        return op.Mul.apply(self, t2)

    # Apply the subtraction operation to the current object and one other tensor.
    def sub(
            self, 
            t2: Tensor
        ) -> Tensor:
        # Validate the device, data type, and shapes.
        self._device_check(t2)
        self._type_check(t2)
        self._shape_check(t2)

        return op.Sub.apply(self, t2)


    # ------- Memory Alteration Ops ------- #

    # Cast a Tensor to a new data type.
    def cast(
            self, 
            new_dtype: DType
        ) -> Tensor:
        # Return the same tensor if the new data type is identical to the current one.
        if self.dtype is new_dtype:
            return self

        # The number of bytes corresponding to a single unit of a data type.
        DTYPE_BYTES: Dict[DType, int] = {
                                        DType.float32: 4, 
                                        DType.int32: 4
                                    }

        # Return the cast tensor.
        return Tensor(self.tdata.cast(new_dtype), new_dtype, self.device.standard(), self.req_grad)


    # Apply a reshape to the tensor; if stride is give, contiguity and memory region size are maintained if the safe flag is active.
    #### NOTE: There is no copy involved here.
    def reshape(
            self, 
            new_shape: List[int], 
            safe: bool = True, 
            new_stride: Optional[List[int]] = None
        ) -> Tensor:
        # If safety is desired, then calculate the number of elements in the provided tensor shape and verify equal memory region size.
        if safe:
            new_ele_cnt: int = functools.reduce(operator.mul, new_shape)
            assert new_ele_cnt == functools.reduce(operator.mul, self.tdata.shape), \
                "The specified reshape cannot be performed as it results in a different sized region of memory than the original allocation."

        # If a stride is provided, then verify that the provided shape and stride have an equal number of dimensions.
        if new_stride:
            assert len(new_shape) == len(new_stride), \
                "The specified reshape cannot be performed because the number of dimensions provided for the shape and stride do not match."

        return op.Reshape.apply(self, new_shape = new_shape, new_stride = new_stride)


    # ------- Memory Movement Ops ------- #

    # Create a contiguous copy of a tensor on the same device.
    def contiguous(
            self
        ) -> Tensor:
        # If the tensor is currently contiguous, then just return the current tensor.
        if self.tdata._contiguous:
            return self

        # Return a contiguous copy of the Tensor.
        return Tensor(self.tdata.contiguous(), self.dtype, self.device.standard(), self.req_grad)

    # Copy a tensor from a source device to a destination device.
    def copy_to(
            self, 
            destination: Device
        ) -> Tensor:
        # If the source and destination devices are the same, then return the current tensor.
        if self.device.standard() == destination.standard():
            return self

        # Create and return a copy of the tensor on the new, specified device.
        return Tensor(self.tdata.copy(destination), self.dtype, destination.standard(), self.req_grad)


    # ------- Reduction Ops ------- #

    # Apply the maximum operation to the current object along a specified axis (and choose whether or not the dimension is preserved).
    def max(
            self, 
            axis: int, 
            keep_dim: bool = False
        ) -> Tensor:
        # Verify that the provided axis is within the dimensional boundary of the tensor and get the new shape of the result.
        new_shape: List[int] = self._reduce_util(axis, keep_dim)

        return op.Max.apply(self, new_shape = new_shape, axis = axis, keep_dim = keep_dim)

    # Apply the minimum operation to the current object along a specified axis (and choose whether or not the dimension is preserved).
    def min(
            self, 
            axis: int, 
            keep_dim: bool = False
        ) -> Tensor:
        # Verify that the provided axis is within the dimensional boundary of the tensor and get the new shape of the result.
        new_shape: List[int] = self._reduce_util(axis, keep_dim)

        return op.Min.apply(self, new_shape = new_shape, axis = axis, keep_dim = keep_dim)

    # Apply the summation operation to the current object along a specified axis (and choose whether or not the dimension is preserved).
    def sum(
            self, 
            axis: int, 
            keep_dim: bool = False
        ) -> Tensor:
        # Verify that the provided axis is within the dimensional boundary of the tensor and get the new shape of the result.
        new_shape: List[int] = self._reduce_util(axis, keep_dim)

        return op.Sum.apply(self, new_shape = new_shape, axis = axis, keep_dim = keep_dim)


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