from copy import deepcopy
import functools
import operator
from dmap.tensor import Tensor
from dmap.ops import Op, Reduce, Binary, Movement, Memory, Fusion
from dmap.ir import IR, MuOp


class Compiler():
    def __init__(self, head: Tensor, fuse: bool = True) -> None:
        self.tokens = self._topological_sort(head) # If the item is returned from a function, then the user
        # should know what the item is by looking at the return type of the function. This means that I should 
        # not need to declare the type here.
        self.ast = [self.lower(token) for token in self.tokens]


    # As the optimier cycles through optimizations, the change is an alternate colored text while the rest is uniform.
    
    # LOAD, TEMP, STORE are almost implicit from ordering/op type.
        
    # First cycle is just the topological sort, then next could be fusion optimizations.

    def _topological_sort(self, tensor: Tensor) -> list[tuple[Op, Tensor, list[Tensor]]]:
        def top_sort_util(tensor, visited, stack) -> list[Tensor]:
            visited.add(tensor)
            for parent in tensor.parents:
                if parent not in visited:
                    top_sort_util(parent, visited, stack)
            if tensor.op not in visited:
                visited.add(tensor.op)
                stack.append((tensor.op, tensor, tensor.parents))
            return stack
        return top_sort_util(tensor, set(), [])
    
    # Thinking about maintaining a similar structure to the previous version. However, am considering
    # adding a mix()/meld/fuse/bind function that somehow draws connections between like operations (mapping 
    # like-axes together and also being able to translate dimensions between off-axes)
    # TEMP was added to ctx below in order to permit this fusion function. We don't want to incur 
    # unnecessary stores when it will just be read back from memory immediately.
    # need to first add the ability to fuse multiple operations together into a fusion token/op

    # Maybe implement limitations if there is an unsafe reshape that has occured between op fusion?
    # Restrict at any reshape? If a reshape happens within fusion and is of a temporary variable, then that 
    # requires conditionally mapping that input without explicitly indexing into it.
    #### I can think through this more in the future.

    # NOTE: I may not even need a fusion function, these can just be labelled as a Fusion Op
    #### These would be created based on logic in the organizational/discovery mechanism and then 
    #### everything operates conditionally based on that.

    ######## This seems to lack a seemless flow. Maybe we predefine the axes in relation to the operation
    ######## by what happens first and then repeatedly run the operations if necessitated by it being
    ######## a fusion operation. I need to think through this a bit more. -> Define dims then map?
    ######## NOTE: If the axes for the entire operation are first defined and then we map, this will make it 
    ######## much easier to perform fusion. We just need to map the intermediate operations (everything before last)
    ######## to temporary. Additionally, if there is an outside tensor in the second operation that does not 
    ######## result from the first, then we need to make sure that it is labelled as a load. Maybe this serves to 
    ######## redefine the context; we should incorporate ordering that alligns with when the tensors are seen...
    ######## i.e. 1: LOAD x, 2: LOAD y, 3: TEMP z, 4: LOAD az, 5: STORE aa
    def lower(self, token: tuple[Op, Tensor, list[Tensor]]) -> list[IR]:
        if isinstance(token[0], Memory) or isinstance(token[0], Movement):
            return [IR(MuOp.NOOP, "", "", [])]
        
        ast: list[IR] = []
        symbol_table: dict[str|Tensor, IR|list[IR]|list[int]] = {}
        ctx: dict[str, list[Tensor]] = {"LOAD": [], "TEMP": [], "STORE": []}