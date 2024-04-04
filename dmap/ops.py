from enum import auto, Enum


class Memory(Enum):
    LOAD = auto()


class Movement(Enum):
    RESHAPE_S = auto()
    RESHAPE_U = auto()


class Unary(Enum):
    EXP = auto()
    LOG = auto()


class Binary(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()


class Reduce(Enum):
    MAX = auto()
    MIN = auto()
    SUM = auto()


class Fusion(Enum):
    ELE_RED = auto()