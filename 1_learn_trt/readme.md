# TensorRT

> 测试平台: i9-9900K + 2080Ti + 32G + Ubuntu18.04 + cuda10.2 + cudnn8.7 + trt8.5.3

1. `trt_demo`: 使用python api和c++ api进行trt模型转换和推理的demo。
2. `trt_plugin`: trt增加自定义plugin的基本demo，以及yolox的nms过程采用trt nms plugin的使用方法。
3. `trt_yolox`: 采用python和c++ 推理yolox的demo。
4. `trt_rtdetr`: paddlepaddle版本rtdetr的trt转换和推理。
