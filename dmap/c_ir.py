from __future__ import annotations
from dmap.first_pass import IR


def to_c_ir(kernel: list[IR]) -> list[IR]:
    for ir in kernel:
        if ir.op == "N-D" or ir.op == "N-R":
            ir.op = "FOR"
            close_loop_to_ir(ir, kernel)
    
    return kernel


def close_loop_to_ir(loop: IR, kernel: list[IR]) -> None:
    loop_set: set[IR] = set([loop])
    delta_len: int = len(loop_set)
    while delta_len != 0:
        start_len: int = len(loop_set)
        for ir in kernel:
            if loop_set.intersection([i for i in ir.dependencies]):
                loop_set.add(ir)
        delta_len = len(loop_set) - start_len
    insert_index: int = max([kernel.index(i) for i in loop_set]) + 1
    temp_end: IR = IR("END", "", "", [loop])
    kernel.insert(insert_index, temp_end)