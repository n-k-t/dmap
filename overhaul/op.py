from __future__ import annotations
from typing import List, Optional

from lazy import LazyTensor
from identifiers import Binary, MemoryAlter, MemoryMove, Reduce, Unary
from tensor import Op


# ------- Binary Ops ------- #

# Create an instance of a tensor addition operation.
class Add(Op):
    def forward(
            self, 
            t1: LazyTensor, 
            t2: LazyTensor 
        ) -> LazyTensor:
        return LazyTensor(t1.shape, t1.dtype, t1.device, parents = [t1, t2], src_op = Binary.ADD)

# Create an instance of a tensor division operation.
class Div(Op):
    def forward(
            self, 
            t1: LazyTensor, 
            t2: LazyTensor 
        ) -> LazyTensor:
        return LazyTensor(t1.shape, t1.dtype, t1.device, parents = [t1, t2], src_op = Binary.DIV)

# Create an instance of a tensor multiplication operation.
class Mul(Op):
    def forward(
            self, 
            t1: LazyTensor, 
            t2: LazyTensor 
        ) -> LazyTensor:
        return LazyTensor(t1.shape, t1.dtype, t1.device, parents = [t1, t2], src_op = Binary.MUL)

# Create an instance of a tensor subtraction operation.
class Sub(Op):
    def forward(
            self, 
            t1: LazyTensor, 
            t2: LazyTensor 
        ) -> LazyTensor:
        return LazyTensor(t1.shape, t1.dtype, t1.device, parents = [t1, t2], src_op = Binary.SUB)


# ------- Memory Alteration Ops ------- #  

# Create an instance of an tensor reshape operation (can be safe or unsafe).
class Reshape(Op):
    def forward(
            self, 
            t1: LazyTensor, 
            new_shape: List[int], 
            new_stride: Optional[List[int]]
        ) -> LazyTensor:
        # Tracking instance information for the backward pass.
        self.new_shape = new_shape
        self.new_stride = new_stride

        return LazyTensor(new_shape, t1.dtype, t1.device, stride = new_stride, parents = [t1], src_op = MemoryAlter.RESHAPE, memory = t1.memory)


# ------- Memory Movement Ops ------- #

# Create an instance of a tensor instantiation operation.
class Instantiate(Op):
    def forward(
            self,
            t1: LazyTensor
        ) -> LazyTensor:
        # A new lazy tensor with identical attributes and memory (provides a load context barrier for backpropagation).
        return LazyTensor(t1.shape, t1.dtype, t1.device, parents = [t1], src_op = MemoryMove.LOAD, memory = t1.memory)


# ------- Reduction Ops ------- #

# Create an instance of a tensor maximum operation (along a specified axis and whether or not the dimension was maintained)
class Max(Op):
    def forward(
            self, 
            t1: LazyTensor, 
            new_shape: List[int], 
            axis: int, 
            keep_dim: bool
        ) -> LazyTensor:
        # Tracking instance information for the backward pass.
        self.axis = axis
        self.keep_dim = keep_dim

        return LazyTensor(new_shape, t1.dtype, t1.device, parents = [t1], src_op = Reduce.MAX, extra = axis)

# Create an instance of a tensor minimum operation (along a specified axis and whether or not the dimension was maintained)
class Min(Op):
    def forward(
            self, 
            t1: LazyTensor, 
            new_shape: List[int], 
            axis: int, 
            keep_dim: bool
        ) -> LazyTensor:
        # Tracking instance information for the backward pass.
        self.axis = axis
        self.keep_dim = keep_dim

        return LazyTensor(new_shape, t1.dtype, t1.device, parents = [t1], src_op = Reduce.MIN, extra = axis)

# Create an instance of a tensor summation operation (along a specified axis and whether or not the dimension was maintained)
class Sum(Op):
    def forward(
            self, 
            t1: LazyTensor, 
            new_shape: List[int], 
            axis: int, 
            keep_dim: bool
        ) -> LazyTensor:
        # Tracking instance information for the backward pass.
        self.axis = axis
        self.keep_dim = keep_dim

        return LazyTensor(new_shape, t1.dtype, t1.device, parents = [t1], src_op = Reduce.SUM, extra = axis)


# ------- Unary Ops ------- #

# Create an instance of a tensor exponentiation operation (base tensor, exponent e).
class Exp(Op):
    def forward(
            self, 
            t1: LazyTensor
        ) -> LazyTensor:
        return LazyTensor(t1.shape, t1.dtype, t1.device, parents = [t1], src_op = Unary.EXP)

# Create an instance of a tensor logarithm operation (base e).
class Log(Op):
    def forward(
            self, 
            t1: LazyTensor
        ) -> LazyTensor:
        return LazyTensor(t1.shape, t1.dtype, t1.device, parents = [t1], src_op = Unary.LOG)