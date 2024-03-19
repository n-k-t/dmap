import ctypes
from dmap.codegen import Code
from dmap.first_pass import IR, Parser
# I think that this will create the code, compile it into a program, then 
# run it, and time it.
# [hash(time.time_ns()) % 21 / 8 for _ in range(1000000)]