from __future__ import annotations
from ops import MemoryOp
from tensor import Tensor

class IR:
    def __init__(
            self,
            op: str,
            data_type: str,
            value: int | float | None = None,
            dependencies: list[IR] = []
        ) -> None:
        self.op: str = op
        self.data_type: str = data_type
        self.value: int | float | None = value
        self.dependencies: list[IR] = dependencies

def to_ir(end: Tensor) -> list[IR]:
    kernel_ir: list[IR] = []
    tensor_list: list[Tensor] = end._topological_sort()

    tensor_pointers: dict[Tensor, IR] = {}
    for num, i in enumerate(tensor_list):
        if isinstance(i._op, MemoryOp):
            temp: IR = IR(op = "ARG", data_type = "float*", value = f"tensor_{num}", dependencies = [])
            kernel_ir.append(temp)
            tensor_pointers[i] = temp
        elif num == len(tensor_list) - 1:
            temp: IR = IR(op = "ARG", data_type = "float*", value = f"tensor_out", dependencies = [])
            kernel_ir.append(temp)
            tensor_pointers[i] = temp

    
    
    return kernel_ir