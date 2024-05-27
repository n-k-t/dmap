from __future__ import annotations
from typing import List, Optional, Union

from device import Device
from lazy import LazyTensor
from static import DType


# The class from which all operation classes inherit their initialization and application methods.
class Op():
    def __init__(
            self, 
            *tensors: Tensor
        ) -> Op:
        self.parents = tensors

    @classmethod
    def apply(
            self, 
            *tensors: Tensor
        ) -> Tensor:
        # Ensure that all tensors being operated upon are on the same device.
        if len(tensors) > 1:
            op_device: str = tensors[0].device.standard()
            assert all(t.device.standard() == op_device for t in tensors), \
                "The tensors provided for the operation must all be on the same device."

        # Instantiate the operation and set the operand tensors.
        op_node: Op = self(*tensors)

        # Get the resulting lazy tensor from the forwards pass.
        res_lazy_t: LazyTensor = op_node.forward(*tensors)

        # Create the resulting tensor and add its source operation (for the backwards pass).
        new_tensor: Tensor = Tensor(res_lazy_t, res_lazy_t.dtype, op_device)
        new_tensor._src_op = op_node

        return new_tensor


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

        # Associate the tensor with either the specified or the base device.
        if device:
            self.device = Device(device)
        else:
            self.device = Device()

        # If the tensor data is not a lazy tensor instance, then convert it into one.
        if isinstance(tdata, LazyTensor):
            self.tdata = tdata
        elif isinstance(tdata, List):
            self.tdata = LazyTensor(tdata, dtype, self.device)

        # Create a hidden "source operation" field that can be filled (for tracing) if an operation is applied.
        self._src_op: Optional[Op] = None