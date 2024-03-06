from __future__ import annotations

# Valid memory operations include the following:
#### LOAD
class MemoryOp:
    def __init__(
            self
        ) -> None:
        self.op: str = 'LOAD'

# Valid movement operations include the following:
#### SAFE_RESHAPE
#### UNSAFE_RESHAPE
class MovementOp:
    def __init__(
            self, 
            op: str, 
            view: list[int],
            stride: list[int] | None = None
        ) -> None:
        self.op: str = op
        self.view: list[int] | None = view
        self.stride: list[int] | None = stride

# Valid unary operations include the following:
#### ABSOLUTE_VALUE
#### LOGARITHM
class UnaryOp:
    def __init__(
            self, 
            op: str
        ) -> None:
        self.op: str = op

# Valid binary operations include the following:
#### ADDITION
#### DIVISION
#### MULTIPLICATION
#### SUBTRACTION
class BinaryOp:
    def __init__(
            self, 
            op: str
        ) -> None:
        self.op: str = op

# Valid reduction operations include the following:
#### MAXIMUM
#### MINIMUM
#### SUMMATION
class ReduceOp:
    def __init__(
            self, 
            op: str, 
            axis: int
        ) -> None:
        self.op: str = op
        self.axis: int = axis