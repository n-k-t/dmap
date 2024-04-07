from __future__ import annotations
from enum import auto, Enum

class MuOp(Enum):
    ARG = auto()
    LOAD = auto()
    STORE = auto()
    CONST = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    N_D = auto()
    N_R = auto()
    LOOP = auto()
    END = auto()
    SUM = auto()
    MAX = auto()
    MIN = auto()
    TEMP = auto()
    EXP = auto()
    LOG = auto()

    def __repr__(self) -> str:
        return str(self)

class IR:
    def __init__(self, op: MuOp, dtype: str, value: str, deps: list[IR]) -> None:
        self.op = op
        self.dtype = dtype
        self.value = value
        self.deps = deps

    def __repr__(self) -> str:
        return " | ".join([
                            f"{self.op:<12}", 
                            f"{self.dtype:<7}", 
                            f"{self.value:<10}", 
                            f"{[(dep.op, dep.value) for dep in self.deps]}"
                        ])    