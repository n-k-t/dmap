from __future__ import annotations

from lazy import LazyTensor
from static import Unary
from tensor import Op


# ------- Unary Ops ------- #

class Add(Op):
    def forward(
            self, 
            t1: LazyTensor, 
            t2: LazyTensor 
        ) -> LazyTensor:
        # NOTE: Add in parents and op (static) for the lazy tensor
        # Do backprop after forward, think about what info is necessary to store
        pass
