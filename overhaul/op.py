from __future__ import annotations
from typing import List

from lazy import LazyTensor
from identifiers import Binary, MemoryAlter, Unary
from tensor import Op


# ------- Binary Ops ------- #

# NOTE: Do backprop after forward, think about what info is necessary to store


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

# Create an instance of a safe tensor reshape operation.
class SafeReshape(Op):
    def forward(
            self, 
            t1: LazyTensor, 
            new_shape: List[int]
        ) -> LazyTensor:
        return LazyTensor(new_shape, t1.dtype, t1.device, parents = [t1], src_op = MemoryAlter.RESHAPE_S, memory = t1.memory)    

# Create an instance of an unsafe tensor reshape operation.
class UnafeReshape(Op):
    def forward(
            self, 
            t1: LazyTensor, 
            new_shape: List[int], 
            new_stride: List[int]
        ) -> LazyTensor:
        return LazyTensor(new_shape, t1.dtype, t1.device, stride = new_stride, parents = [t1], src_op = MemoryAlter.RESHAPE_U, memory = t1.memory)

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