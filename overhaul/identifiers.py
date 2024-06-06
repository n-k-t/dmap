from enum import auto, Enum


# An enum representing a binary operation of a tensor.
class Binary(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()

# An enum representing the data type of a tensor.
class DType(Enum):
    float32 = auto()
    int32 = auto()

# An enum representing a fusion of multiple tensors' operations.
class Fusion(Enum):
    ELE_RED = auto()

# An enum representing a memory altering operation for a tensor.
class MemoryAlter(Enum):
    RESHAPE_S = auto()
    RESHAPE_U = auto()

# An enum representing a memory movement operation for a tensor.
class MemoryMove(Enum):
    LOAD = auto()

# An enum representing a reduction operation of a tensor.
class Reduce(Enum):
    MAX = auto()
    MIN = auto()
    SUM = auto()

# An enum representing a unary operation of a tensor.
class Unary(Enum):
    EXP = auto()
    LOG = auto()