from __future__ import annotations
from enum import auto, Enum

class MuOp(Enum):
    LOAD = auto()
    STORE = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    LOOP = auto()
    END = auto()
    IF = auto()
    CMPR = auto()
    PHI = auto()
    NOOP = auto()

class IR:
    def __init__(self, op: MuOp, dtype: str, value: str, deps: list[IR]) -> None:
        self.op = op
        self.dtype = dtype
        self.value = value
        self.deps = deps

    def __repr__(self) -> str:
        return "\t".join([
                            f"OP: {self.op:>10},", 
                            f"DT: {self.dtype:>10},", 
                            f"VAL: {self.value:>10},", 
                            f"DEP: {[(dep.op, dep.value) for dep in self.deps]}"
                        ])    