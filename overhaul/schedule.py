# Should I linearize the schedule, creating intermediate operations along the way?
# Could make it more difficult to figure out where we store vs where we use registers/shared memory

# What about going through and placing tensors as load/store as encountered
# If the parent op of a tensor is encountered, then it can be moved from load to 
# store (or temp if we don't need it for later --> training)
# Could be a BFS, but should think about how to do this, don't want huge spidering operations
# Should I track the degree of a node before adding
# Also, how to pattern match for fusion?



# Currently have a map of memory buffers
# Go through this and convert it a map of lazy tensor ops
#### NOTE: Keep track of relative level (as list if used multiple times???)
# Make sure this is connected and we can get the degree of each node
# Can then fuse these together in a fusion lazy tensor op based on patterns or criteria
# Can traverse the op graphs and decide how to store the results in memory/their operations

from __future__ import annotations
from collections import deque
from typing import Deque, Dict, List, Tuple, Union

from identifiers import Binary, MemoryAlter, MemoryMove, Reduce, Unary
from lazy import LazyTensor


class LazyOp():
    def __init__(
            self, 
            op: Union[Binary, MemoryAlter, MemoryMove, Reduce, Unary], 
            srcs: Tuple[LazyOp], 
            in_t: List[LazyTensor], 
            out_t: LazyTensor, 
            in_degree: int = 0
        ) -> LazyOp:
        self.op = op
        self.srcs = srcs
        self.in_t = in_t
        self.out_t = out_t

        # Track the number of children that an operation has (defaults to zero).
        self.in_degree = in_degree


# Perform breadth-first search of the LazyTensor graph, transforming it into a LazyOp graph.
def lt_to_lo(
        branch: LazyTensor
    ) -> LazyOp:
    # Create the initial LazyOp for the LazyTensor passed in.
    l_op: LazyOp = LazyOp(branch.src_op, (), branch.parents, branch)

    # Create queues for both the LazyTensors and LazyOps; the order is identical between the two.
    #### NOTE: A deque is used for easy FIFO (first in, first out -> pop left).
    lt_queue: Deque = deque([branch])
    lf_queue: Deque = deque([l_op])

    # Track the LazyOp where a LazyTensor is the output (prevents duplication of operations).
    lt_out_dict: Dict[LazyTensor, LazyOp] = {branch: l_op}

    # Iterate until a queue is empty.
    while len(lt_queue) > 0:
        # Pop the first element of the LazyTensor and LazyOp queues.
        lt_node: LazyTensor = lt_queue.popleft()
        lf_node: LazyOp = lf_queue.popleft()

        # Iterate over the parent LazyTensors of the current LazyTensor.
        for parent in lt_node.parents:
            # Create a new LazyOp if the parent is not in the tracking dict, else increase the in-degree.
            if parent not in lt_out_dict:
                temp_l_op: LazyOp = LazyOp(parent.src_op, (), parent.parents, parent, 1)

                # Add the new LazyOp as a source LazyOp for the one that was popped this iteration.
                lf_node.srcs += (temp_l_op, )

                # Add the LazyTensor and LazyOp to the queues.
                lt_queue.append(parent)
                lf_queue.append(temp_l_op)

                # Create a dictionary entry where the LazyTensor is the key and the LazyOp is the value.
                lt_out_dict[parent] = temp_l_op
            else:
                # Add the pre-created LazyOp as a source to the LazyOp that was popped this iteration.
                lf_node.srcs += (lt_out_dict[parent], )

                # Increase the in-degree of the pre-created Lazy-Op.
                lt_out_dict[parent].in_degree += 1

    return l_op


class Kernel():
    def __init__(
            self, 
            ast: List[LazyOp], 
            in_t: List[LazyTensor], 
            out_t: List[LazyTensor]
        ) -> Kernel:
        self.ast = ast
        self.in_t = in_t
        self.out_t = out_t

# Need a schedule item class or something to contain everything
# Should this group together preliminary ops for loops or no?

# NOTE: an evaluated tensor means that it will be either an input or an output for an operation.
# NOTE: We check for evaluated or a LOAD operation
# We can make some patterns/ rules to decide what else is evaluated
# Do I flatten this out and we keep track of i.e. in-degree (this translates to how many kinds the result has), etc.

# NOTE: We're assuming everything is happening on the same device. I should check that.
#### If something is happening on a separate device, then there needs to be a copy operation.

# TODO: Now should be ready to start separating into kernels, HEED the advice I've placed around here.


# NOTE: If we start sectioning things here, will need to track multiple "branches"
# as we make cuts. This is only if we start sectioning the tree into disjoint subtrees.