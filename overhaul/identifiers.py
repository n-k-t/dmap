from enum import auto, Enum


# ------- Tensor Data Types ------- #

# An enum representing the data type of a tensor.
class DType(Enum):
    float32 = auto()
    int32 = auto()


# ------- Tensor Operations ------- #

# An enum representing a binary operation of a tensor.
class Binary(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()

# An enum representing a memory altering operation for a tensor.
class MemoryAlter(Enum):
    CAST = auto()
    RESHAPE = auto()

# An enum representing a memory movement operation for a tensor.
class MemoryMove(Enum):
    CONTIGUOUS = auto()
    COPY = auto()
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


# ------- Tensor Fusion Operations ------- #

# An enum representing a fusion of multiple tensors' LazyOps.
class Fusion(Enum):
    ELE_RED = auto()
    MUL_ACC = auto()