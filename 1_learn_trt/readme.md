# TensorRT

这是一个tensorRT的仓库。测试平台是：
> i9-9900K + 2080Ti + 32G + Ubuntu18.04 + cuda10.2 + cudnn8.7 + trt8.5.3

1. `trt_demo`: 几个trt运行的demo，模型转换之类的。
2. `trt_plugin`: trt增加plugin的基本demo，以及yolo融合nms后处理
3. `trt_yolox`: yolox推理的py和cpp demo
4. `trt_rtdetr`: rtdetr trt