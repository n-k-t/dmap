from enum import auto, Enum
from dmap.tensor import Tensor

class Muop(Enum):
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

class IR:
    def __init__(self, op: Muop, dtype: str, value: str, deps: list[IR]) -> None:
        self.op = op
        self.dtype = dtype
        self.value = value
        self.deps = deps

    # Can we make this all one line with join?
    def __repr__(self) -> str:
        return f"OP: {self.op:>10},\tDT: {self.dtype:>10},\tVAL: {self.value:>10},\tDEP: {[(dep.op, dep.value) for dep in self.deps]}"
    