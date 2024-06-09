# Can I make a very simple graph and then all optimizations and lowering is pattern matchins MuOps???

# For specific optimizations, create pattern latches (or magnets) that force that pattern to be evaluated first?


from __future__ import annotations
from typing import Generator, List, Optional
import functools
import operator

from identifiers import Binary, Unary, Reduce
from ir import IR, MuOp
from lazy import LazyTensor
from schedule import Event, LazyOp

# Do I begin by linearizing the subtree and finding max dims of the problem (for all subtrees)
# Track the reduce dimensions and put them in the center
# Function to find the possible optimizations?
# track the device

# Optimizations are not fed in, instead they are discovered by a function and can be iterated over
# Different for CPU vs GPU
# ALL: Tesselate -> Permute -> Unroll
# GPU Only: Globals and Locals

# Order for checking is tesselate, permute, g&l, unroll
# SIMD at the same time as tesselate??? Does this include unroll as well?

# If contiguous can add in a SIMD option too
# NOTE: Should add in conditionals for the optimizations (i.e. non-integer tesselation)
# TODO: Tensor Cores are the only optimization that will allow for non-integer tesselation.

# Optimizations get their own file (should need to import ir and device)

# First pass just creates a VERY simple case
#### ADD (ARG1, ARG2) (no strides yet etc)
# ARGS will have their dependencies stored so we know where everything points to (maybe in stride order too?)
# We label the dims and everything and can also have loads/stores as well as temps
# Then, we check for opts as above
# These can be applied easily (i.e. tesselate -> insert new dimension and add it as dependency for all children that need it)
# If we unroll, can add RANGE, this will act as a mark that we need to repeatedly create the sequence of ALL MuOps below it

# Have a separate IR called constructors that hold the N_D, N_R, RANGE (Ops)
# NOTE: Could have their own operands for now connecting everything?
# TODO: Keep track of where in loops everything goes by keeping dependencies in separate lists
#### Thinking about ways to easily move axes AND their dependencies
#### If each (OP, not at tensor level) is tied to their respective axes, then simply in the ordered loops it must be at the max dep level
# Do I need stores and loads in the construct phase?
# ConstructIR serve as the framework for building the MuOp IR, should make it much easier to try many optimizations instead of 
# building from scratch every single time.

# Don't need a cast constructor, just make a list of ops to be cast and check when building
# Keep a list of the dimensions as well
# Each tensor pointer in the constructors also points to each of the axes it is dependent on
# How do we find the parent/children if we have many subtrees???
#### Do we flatten everything to make it an easier search?
#### Or because it's all in a list, we just check each subtree?
# RANGE is not applied until any ops are applied, otherwise everythin can be loaded as usual.

class Kernel():
    def __init__(
            self, 
            event: Event, 
            opt: bool = False
        ) -> Kernel:
        self.event = event
        self.in_t: Optional[List[LazyTensor]] = None
        self.out_t: Optional[List[LazyTensor]] = None
        self.num_flops: int = 0


    # ------- Utility Functions ------- #

    # Update the FLOP (floating point operation) tracker for the kernel.
    def _calc_flops(
            self, 
            ops: List[LazyOp]
        ) -> None:
        flop_tracker: int = 0

        # Iterate over each operation and update the flop tracker accordingly.
        #### NOTE: Binary/unary ops are the volume of the operation.
        #### NOTE: Reduce ops are the volume with the reduction axis decremented by one.
        for op in ops:
            if (isinstance(op.op, Binary)) or (isinstance(op.op, Unary)):
                flop_tracker += functools.reduce(operator.mul, op.out_t.shape)
            elif isinstance(op.op, Reduce):
                adj_shape: Generator[int] = (i if num != op.extra else i - 1 for num, i in enumerate(op.out_t.shape))
                flop_tracker += functools.reduce(operator.mul, adj_shape)

        self.num_flops += flop_tracker


    # ------- First Pass ------- #

    def lower(
            self, 
            event: Event, 
            opt: bool
        ) -> List[MuOp]:

        self.in_t = event.in_t
        self.out_t = event.out_t

        def util(
                l_o: LazyOp
            ) -> Generator[LazyOp]:
            for num, i in enumerate(l_o.barriers):
                if not i:
                    yield from util(l_o.srcs[num])

            yield l_o
        
        flattened: List[MuOp] = list(util(event.ops[0]))

        self._calc_flops(flattened)
