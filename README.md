# Faster Deployment

本仓库主要是深度学习模型的TensorRT和ncnn部署，有较好的接口便捷性和推理延迟。均以目标检测部署为例。
> 另外需要说明的是，不同于其他benchmark，我的项目因为主要适用于机器人上的，所以在测试的时候不会采用`img`或者`batch img`进行测试，均是在`batch=1`时候，一张接着一张图片输入或者视频输入的方式进行测试，模拟单目机器人的实际情况。

- trt_learn: 主要是tensorrt的基础操作，模型转换 推理 插件等

## 致谢
感谢手写ai团队开源的[TensorRT_Pro](https://github.com/shouxieai/tensorRT_Pro)，让我受益良多，本仓库中tensorrt的代码也均是在其基础上面进行小部分优化，以及按照该仓库的整体思路优化ncnn的推理。




