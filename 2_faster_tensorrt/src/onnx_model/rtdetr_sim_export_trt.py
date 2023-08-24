import os
import onnx_graphsurgeon as gs
import onnx

# paddle上的修改可以参考这位大佬的文章
# https://zhuanlan.zhihu.com/p/623794029

model = onnx.load("./rtdetr_r18vd_6x_coco.onnx")
graph = gs.import_onnx(model)
graph.outputs[0].name = "output"
# print(graph.outputs)

onnx.save(gs.export_onnx(graph), "rtdetr_r18vd_6x_coco_output.onnx")

os.system("onnxsim rtdetr_r18vd_6x_coco_output.onnx rtdetr_r18vd_6x_coco_output_sim.onnx")

os.system("trtexec --onnx=./rtdetr_r18vd_6x_coco_output_sim.onnx --workspace=4096 --shapes=image:1x3x640x640 --saveEngine=rtdetr_r18vd_6x_coco.trt  --fp16")

