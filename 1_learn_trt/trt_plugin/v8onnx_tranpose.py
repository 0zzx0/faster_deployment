import onnx
import numpy as np
import onnx_graphsurgeon as gs


"""
pip install nvidia-pyindex
pip install onnx-graphsurgeon 

"""

model = onnx.load("./onnx_weights/v8n.onnx")
graph = gs.import_onnx(model)

# graph.outputs[0].name = "output"
old_shape = graph.outputs[0].shape
output_tensort = gs.Variable("output", graph.outputs[0].dtype, [old_shape[0], old_shape[2], old_shape[1]] )

graph.nodes[-1].outputs[0].name = "oldoutput"

reshape_node = gs.Node(
    op="Transpose",
    name="outputtranspose",
    inputs=[graph.nodes[-1].outputs[0]],
    outputs=[output_tensort],
    attrs={"perm": [0, 2, 1]}
)

# print(type(graph.nodes)) # list
graph.nodes.append(reshape_node)


graph.outputs = reshape_node.outputs
for node in graph.outputs:
    print(node)


graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "v8_transpose.onnx")

