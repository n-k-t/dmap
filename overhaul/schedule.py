from __future__ import annotations
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple, Union

from identifiers import Binary, MemoryAlter, MemoryMove, Reduce, Unary
from lazy import LazyTensor


# A class that transforms the lazy focus from tensor-oriented to operation oriented.
class LazyOp():
    def __init__(
            self, 
            op: Union[Binary, MemoryAlter, MemoryMove, Reduce, Unary], 
            srcs: Tuple[LazyOp], 
            in_t: List[LazyTensor], 
            out_t: LazyTensor, 
            in_degree: int = 0, 
            extra: Optional[int] = None
        ) -> LazyOp:
        self.op = op
        self.srcs = srcs
        self.in_t = in_t
        self.out_t = out_t

        # Track the number of children that an operation has (defaults to zero).
        self.in_degree = in_degree

        # Store extra information (i.e. reduce axis).
        self.extra = extra

        # Indicates whether or not the this is a terminal node of the subtree for a specific branch.
        self.barriers: List[bool] = []


# A class that represents an event in the schedule (this will become a kernel).
class Event():
    def __init__(
            self, 
            ops: List[LazyOp], 
            in_t: List[LazyTensor], 
            out_t: List[LazyTensor]
        ) -> Event:
        self.ops = ops
        self.in_t = in_t
        self.out_t = out_t


# A schedule indicating how to break down and order the execution of the lazy graphs.
class Schedule():
    def __init__(
            self, 
            l_t: LazyTensor
        ) -> Schedule:
        self.events: List[Event] = []

        # Create the schedule upon initialization, first transforming LazyTensors into LazyOps.
        self._make_schedule(self._lt_to_lo(l_t))

        # Reverse the Event order because it was created by traversing from the bottom of the LazyOp tree.
        self.events.reverse()


    # ------- LazyTensors -> LazyOps ------- #

    # Perform breadth-first search of the LazyTensor graph, transforming it into a LazyOp graph.
    def _lt_to_lo(
            self, 
            branch: LazyTensor
        ) -> LazyOp:
        # Create the initial LazyOp for the LazyTensor passed in.
        l_op: LazyOp = LazyOp(branch.src_op, (), branch.parents, branch, extra = branch.extra)

        # Create a queue for both the LazyTensors and LazyOps.
        #### NOTE: A deque is used for easy FIFO (first in, first out -> pop left).
        lt_lo_queue: Deque = deque([(branch, l_op)])

        # Track the LazyOp where a LazyTensor is the output (prevents duplication of operations).
        lt_out_dict: Dict[LazyTensor, LazyOp] = {branch: l_op}

        # Iterate until the queue is empty.
        while len(lt_lo_queue) > 0:
            # Pop the first element of the LazyTensor and LazyOp queues.
            lt_node, lo_node = lt_lo_queue.popleft()

            # Iterate over the parent LazyTensors of the current LazyTensor.
            for parent in lt_node.parents:
                # Create a new LazyOp if the parent is not in the tracking dict, else increase the in-degree.
                if parent not in lt_out_dict:
                    temp_l_op: LazyOp = LazyOp(parent.src_op, (), parent.parents, parent, 1, parent.extra)

                    # Add the new LazyOp as a source LazyOp for the one that was popped this iteration.
                    lo_node.srcs += (temp_l_op, )

                    # Add the discovered LazyTensor and LazyOp to the queue.
                    lt_lo_queue.append((parent, temp_l_op))

                    # Create a dictionary entry where the LazyTensor is the key and the LazyOp is the value.
                    lt_out_dict[parent] = temp_l_op
                else:
                    # Add the pre-created LazyOp as a source to the LazyOp that was popped this iteration.
                    lo_node.srcs += (lt_out_dict[parent], )

                    # Increase the in-degree of the pre-created Lazy-Op.
                    lt_out_dict[parent].in_degree += 1

                # Adjust the number of possible barriers to match the number of source branches (this remains zero if there is no parent).
                lo_node.barriers.append(False)

        # Return the root of the LazyOp tree (the branch of the tensor operations).
        return l_op


    # ------- LazyOps -> Events ------- #

    # Create a schedule of root LazyOps and determine if LazyTensors are inputs or outputs.
    def _make_schedule(
            self, 
            branch: LazyOp
        ) -> None:
        # Create a queue to maintain the order in which LazyOps are visited; these are only the tree roots for Event.ops.
        lo_queue: Deque = deque([branch])

        # A loop that will run until all possible LazyOps are reached.
        while len(lo_queue) > 0:
            # Pop the LazyOp at the front of the queue.
            start: LazyOp = lo_queue.popleft()

            # Set the tensor to be evaluated as it will be the output for this scheduled item.
            start.out_t.evaluated = True

            # Create an instance of the event setting the root, input LazyTensors, and output LazyTensor from the popped LazyOp.
            cur_event: Event = Event([start], start.in_t, [start.out_t])

            # Create a queue that will track the ops visited by a path from the Event.ops root currently being iterated over.
            #### NOTE: This will only contain nodes which, when discovered, have a degree of zero (accounting for the discovery).
            traversal_queue: Deque = deque([start])

            # Keep track of the number of reduce operations in a scheduled event (the hard limit is one).
            reduce_count: int = 0

            # Iterate over the LazyOps on the path from the current Event.ops root.
            while len(traversal_queue) > 0:
                # Pop the first LazyOp on the path.
                position: LazyOp = traversal_queue.popleft()

                # Verify that its in-degree is 0.
                if position.in_degree != 0:
                    raise ValueError("A LazyOp with an in-degree not equal to zero cannot be scheduled.")

                # If the LazyOp is a reduction operation, then increment the reduce count by one.
                if isinstance(position.op, Reduce):
                    reduce_count += 1

                # If the source operation is a memory alteration (reshape) or copy (including contiguous) operation, end the loop iteration and start the next on the parent LazyOp.
                if (position.op is MemoryAlter.RESHAPE) or (position.op is MemoryMove.CONTIGUOUS) or (position.op is MemoryMove.COPY):
                    # Set a barrier along the path from the current LazyOp to the source.
                    position.barriers[0] = True

                    # Reduces the in-degree of the source by one.
                    position.srcs[0].in_degree -= 1

                    # Set the LazyTensor output of the source to be evaluated.
                    position.srcs[0].out_t.evaluated = True
                    
                    # It adds it to the LazyOp queue to become a root for an Event.
                    lo_queue.append(position.srcs[0])

                    # Start the next loop iteration.
                    continue

                # Iterate over all of the parent operations of the LazyOp.
                for src_op in position.srcs:
                    # Check that the LazyOp has a degree of 1 (accounting for the discovery) to guarantee no other dependencies.
                    if src_op.in_degree > 1:
                        # If not, then we create a barrier in the current event along that LazyOp path.
                        position.barriers[position.srcs.index(src_op)] = True

                        # Then we decrement the degree by one because we have discovered it along this path.
                        src_op.in_degree -= 1

                        # Add the output of the LazyOp as an input for the event if it is not already included.
                        if src_op.out_t not in cur_event.in_t:
                            cur_event.in_t.append(src_op.out_t)

                        # If the LazyOp is a load operation, then move the next loop iteration (the LazyTensor is already evaluated).
                        if src_op.op is MemoryMove.LOAD:
                            continue

                        # Then we evaluate the LazyTensor as its output must be an input (is a dependency) for the discoverer.
                        src_op.out_t.evaluated = True

                        # Skip to the next iteration of the loop.
                        continue
                    # If the LazyOp is a memory reshape, contiguous copy, or movement (i.e. load/copy), then pass the conditionals.
                    elif (src_op.op is MemoryAlter.RESHAPE) or (position.op is MemoryMove.CONTIGUOUS) or (isinstance(src_op.op, MemoryMove)):
                        pass
                    # If the LazyOp is a reduce operation (when the hard limit is reached), then potentially force the evaluation
                    elif (isinstance(src_op.op, Reduce)) and (reduce_count > 0):
                        # Indicate that the operation was not intended to be evaluated.
                        if src_op.out_t.evaluated == False:
                            src_op.out_t.force_evaluated = True
                    # Only runs if none of the conditions above are executed.
                    else:
                        # Iterate through all input LazyTensors in the parent LazyOp and add as inputs to the event (if not already included).
                        for l_t in src_op.in_t:
                            if l_t not in cur_event.in_t:
                                cur_event.in_t.append(l_t)

                        # If the LazyOp output is in the event input, then remove it.
                        if src_op.out_t in cur_event.in_t:
                            cur_event.in_t.pop(cur_event.in_t.index(src_op.out_t))

                        # Check if the LazyTensor is to be evaluated or not and add it to the output if it should be.
                        if src_op.out_t.evaluated:
                            cur_event.out_t.append(position.out_t)

                        # Decrement the in-degree of the LazyOp by one.
                        src_op.in_degree -= 1

                        # Add the LazyOp to the traversal queue so its connections can be explored.
                        traversal_queue.append(src_op)

                        # Begin the next iteration of the loop.
                        continue

                    # Passing the conditionals creates a barrier along the path from the current to the parent LazyOp.
                    position.barriers[position.srcs.index(src_op)] = True

                    # It reduces the in-degree by one.
                    src_op.in_degree -= 1

                    # It sets the LazyTensor to be evaluated.
                    src_op.out_t.evaluated = True
                    
                    # It adds it to the LazyOp queue to become a root for an Event.
                    lo_queue.append(src_op)

            # Reverse the order of the Event.ops because we started from the bottom of the operation tree.
            cur_event.ops.reverse()

            # Add the current Event to the Schedule.
            self.events.append(cur_event)