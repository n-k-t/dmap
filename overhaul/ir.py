from __future__ import annotations
from enum import Enum, auto
from typing import List, Union

from identifiers import Binary, DType, Reduce, Unary


class ConstructIR(Enum):
    ALU = auto()
    RANGE = auto()
    N_D = auto()
    N_R = auto()


class Constructor():
    def __init__(
            self, 
            op: ConstructIR, 
            value: Union[Binary, Reduce, Unary], 
            deps: List[Constructor]
        ) -> Constructor:
        self.op = op
        self.value = value
        self.deps = deps

    def __repr__(
            self
        ) -> str:
        return " | ".join([
                            f"{self.op:<12}", 
                            f"{self.value:<10}", 
                            f"{[(dep.op, dep.value) for dep in self.deps]}"
                        ])  


class MuOp(Enum):
    ARG = auto()
    LOAD = auto()
    STORE = auto()
    CONST = auto()
    LOOP = auto()
    END = auto()


class IR():
    def __init__(
            self, 
            op: MuOp, 
            dtype: DType, 
            value: str, 
            deps: List[IR]
        ) -> IR:
        self.op = op
        self.dtype = dtype
        self.value = value
        self.deps = deps

    def __repr__(
            self
        ) -> str:
        return " | ".join([
                            f"{self.op:<12}", 
                            f"{self.dtype:<7}", 
                            f"{self.value:<10}", 
                            f"{[(dep.op, dep.value) for dep in self.deps]}"
                        ])  