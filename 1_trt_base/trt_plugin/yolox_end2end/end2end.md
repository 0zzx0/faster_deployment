# Yolo Convert End2End Tensorrt

yolox默认的后处理是在cpu上直接进行了，所以在经过tensorrt加速后其实后处理还是在cpu上进行。于是考虑吧nms的过程加入到tensorrt生成序列化文件的过程中。因为trt已经有了nms的插件，所以加进去就行了，目前主要有两种思路：

1. 参考mmyolo的easydeploy，设置一个TRT::EfficientNMS 的op，在export onnx的时候同时转过去，然后trtexec在模型转换中会自动识别到NMS的插件并替换，实现端到端的操作。

2. 生成序列化文件后，通过增加插件层，来让模型在运行中调用NMS插件，实现端到端。

## 0. 一些实验
首先是发现end2end后反而慢了很多再找问题
1. 测试模型转换。利用trtexec和torch2trt导出测试结果合速度都基本一致。排除此问题。非端到端处理的情况下运行速度为2.84-2.85ms
2. 经过测试，发现转成端到端的模型后，在c++环境下有加速效果5.045ms->4.577ms. 但是在py环境下反而速度降低

下面是当前的测试结果（通过增加op的方法），处理时间包括前处理（除imread以外的所有操作），后处理（包括nms，不包括绘制矩形和保存图片）。程序首先预热50轮，然后连续运行1000轮，计算平均耗时。
> 平台: 2080ti + i9-9900K + trt8.5 + cuda10.2 + cudnn8.7

| 程序      | normal    | end2end |            
| :-----:   | :-----:   | :-----: |                                   
| python    | 4.00ms   | 3.26ms |    
| C++       | 5.045ms   | 4.577ms |    


这里看到c++要不py慢这么多，经过试验发现主要是前处理的差距太大了。如果不包括前处理阶段,这样就合理很多了
| 程序         | end2end |            
| :-----:     | :-----: |                                   
| python      | 1.94ms |    
| C++         | 1.8ms |  

## 1. 增加OP
首先定义一个文件，作为转onnx的过渡op
```python
import torch
from torch import Tensor
# refer https://github.com/open-mmlab/mmyolo/blob/dev/projects/easydeploy/nms/trt_nms.py

# onnx 自定义节点
class TRTEfficientNMSop(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        boxes: Tensor,
        scores: Tensor,
        background_class: int = -1,
        box_coding: int = 0,
        iou_threshold: float = 0.45,
        max_output_boxes: int = 100,
        plugin_version: str = '1',
        score_activation: int = 0,
        score_threshold: float = 0.25,
    ):
        batch_size, _, num_classes = scores.shape
        num_det = torch.randint(
            0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(
            0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(g,
                 boxes: Tensor,
                 scores: Tensor,
                 background_class: int = -1,
                 box_coding: int = 0,
                 iou_threshold: float = 0.45,
                 max_output_boxes: int = 100,
                 plugin_version: str = '1',
                 score_activation: int = 0,
                 score_threshold: float = 0.25):
        out = g.op(
            'TRT::EfficientNMS_TRT',
            boxes,
            scores,
            background_class_i=background_class,
            box_coding_i=box_coding,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            plugin_version_s=plugin_version,
            score_activation_i=score_activation,
            score_threshold_f=score_threshold,
            outputs=4)
        num_det, det_boxes, det_scores, det_classes = out
        return num_det, det_boxes, det_scores, det_classes


def efficient_nms(
    boxes: Tensor,
    scores: Tensor,
    max_output_boxes_per_class: int = 1000,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    pre_top_k: int = -1,
    keep_top_k: int = 100,
    box_coding: int = 0,
):
  
    num_det, det_boxes, det_scores, det_classes = TRTEfficientNMSop.apply(
        boxes, scores, -1, box_coding, iou_threshold, keep_top_k, '1', 0,
        score_threshold)
    return num_det, det_boxes, det_scores, det_classes

```

针对yolox输出的模型，建立一个部署模型来增加后处理op，这个地方需要注意nms的输入需要对应到官方插件库的输入输出说明[nms tensorrt-plugin](https://github.com/NVIDIA/TensorRT/tree/release/8.5/plugin/efficientNMSPlugin),对这个新模型进行export即可，
```python
class DeployModel(nn.Module):
    def __init__(self, baseModel: nn.Module):
        super().__init__()
        self.baseModel = baseModel
        
        self.pre_top_k =  1000
        self.keep_top_k =  100
        self.iou_threshold =  0.45
        self.score_threshold =  0.1

    def forward(self, inputs: Tensor):
        outputs = self.baseModel(inputs)

        bboxes = outputs[:, :, :4]
        scores = outputs[:, :,  4:5] * outputs[:, :, 5:]
        
        return efficient_nms(bboxes, scores, self.keep_top_k, self.iou_threshold,
                        self.score_threshold, self.pre_top_k, self.keep_top_k, box_coding=1)
```

然后就可以直接使用trtexec转成trt的engine文件了。

