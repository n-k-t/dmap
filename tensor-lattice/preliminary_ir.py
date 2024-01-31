# Write everything in a similar manner to before (SSA), but it is more general. 
#### This means replace loops/other specific operations in terms of n-nodes and 
#### phi-nodes. I believe that we can maintain the indexing in the same manner.
#### If not, we can create a load tied to an index operation that links to the 
#### specified dimension nodes.

# arg operand_0 --> store the actual Tensors?
# arg operand_1
# arg result

# n-D value_0
# n-D value_1
# n-D value_2

# load op_0 ... --> indices (point to n-D)
# load op_1 ... --> indices
#### Need to add in offsets here if there is one.

# phi path 1 or path 2 (likely built into the load, specifically if there is a mask)
#### Use a phi for every load? points to none if there is no mask present?
#### Also use if there is an offset?
#### Could need "AND" and "OR" nodes as well if multiple things need to be checked
#### This could get complex, should just do 0 padding for now.
#### Look into this for pads then reshapes. I think we will need to check both the % and //
#### for the global dims.

# join_ob op_0 & op_1

# load res ... --> indices
# store join in res

