from tensor import Tensor
from ops import ReduceOp
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph

def plot_tensor_graph(start: Tensor, file_name: str) -> nx.DiGraph:
    G = nx.DiGraph()
    op_colors = {"LOAD": "cyan", "SAFE_RESHAPE": "green", "UNSAFE_RESHAPE": "#ffcccb"}
    queue = [start]
    while len(queue) > 0:
        temp = queue.pop(0)
        if temp not in G:
            G.add_node(temp, style = "filled", label = str(temp._memory.view), fillcolor = "white")
        if str(temp._op.op) in op_colors:
            G.add_node(temp._op, style = "filled", label = str(temp._op.op), fillcolor = op_colors[str(temp._op.op)])
        elif isinstance(temp._op, ReduceOp):
            G.add_node(temp._op, style = "filled", label = str(temp._op.op), fillcolor = "lavender")
        else:
            G.add_node(temp._op, style = "filled", label = str(temp._op.op), fillcolor = "yellow")
        G.add_edge(temp._op, temp)
        if len(temp._parents) > 0:
            for parent in temp._parents:
                G.add_node(parent, style = "filled", label = str(parent._memory.view), fillcolor = "white")
                G.add_edge(parent, temp._op)
                queue.append(parent)
    A = to_agraph(G)
    A.layout('dot')
    A.draw(file_name)