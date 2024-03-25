from __future__ import annotations
from enum import auto, Enum
from typing import Optional

class Op:
    def __init__(self, 
                op: Memory|Movement|Unary|Binary|Reduce, 
                view: Optional[list[int]] = None, 
                stride: Optional[list[int]] = None,
                axis: Optional[int] = None
                ) -> None:
        self.op = op
        self.view = view
        self.stride = stride
        self.axis = axis



# Valid memory operations include the following:
#### LOAD
class Memory(Enum):
    LOAD = auto()
# class MemoryOp:
#     def __init__(
#             self
#         ) -> None:
#         self.op: str = 'LOAD'


# Valid movement operations include the following:
#### SAFE_RESHAPE
#### UNSAFE_RESHAPE
class Movement(Enum):
    RESHAPE_S = auto()
    RESHAPE_U = auto()
# class MovementOp:
#     def __init__(
#             self, 
#             op: str, 
#             view: list[int],
#             stride: list[int] | None = None
#         ) -> None:
#         self.op: str = op
#         self.view: list[int] | None = view
#         self.stride: list[int] | None = stride


# Valid unary operations include the following:
#### EXPONENTIAL
#### LOGARITHM
class Unary(Enum):
    EXP = auto()
    LOG = auto()
# class UnaryOp:
#     def __init__(
#             self, 
#             op: str
#         ) -> None:
#         self.op: str = op


# Valid binary operations include the following:
#### ADDITION
#### DIVISION
#### MULTIPLICATION
#### SUBTRACTION
class Binary(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
# class BinaryOp:
#     def __init__(
#             self, 
#             op: str
#         ) -> None:
#         self.op: str = op


# Valid reduction operations include the following:
#### MAXIMUM
#### MINIMUM
#### SUMMATION
class Reduce(Enum):
    MAX = auto()
    MIN = auto()
    SUM = auto()
# class ReduceOp:
#     def __init__(
#             self, 
#             op: str, 
#             axis: int
#         ) -> None:
#         self.op: str = op
#         self.axis: int = axis