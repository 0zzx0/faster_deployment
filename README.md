# Faster Deployment

本仓库主要是深度学习模型的TensorRT和ncnn部署，有较好的接口便捷性和推理延迟。均以目标检测部署为例。
> 另外需要说明的是，不同于其他benchmark，本项目因为主要适用于机器人，所以在测试是一般设置`batch=1`，采用一张接着一张图片输入或者视频输入的方式进行测试，模拟单目机器人的实际情况。

- 1_trt_learn: 主要是tensorrt的基础操作，模型转换 推理 插件等 以及一些运行的demo
- 2_faster_tensorrt: 主要是tensorrt的封装和优化
- 3_faster_ncnn: 参考`2_faster_tensorr`封装ncnn

## 致谢
首先需要感谢手写ai团队开源的[TensorRT_Pro](https://github.com/shouxieai/tensorRT_Pro)，让我受益良多，本仓库中tensorrt的代码也均是在其基础上面进行小部分优化，以及按照该仓库的整体思路优化ncnn的推理。该仓库的部分优点：
1. 接口简单清晰
2. 预处理和后处理自写CUDA加速
3. batch可根据实际数据动态调整(前提是trtmodel转换中设置动态batch)
4. 写了内存和数据的管理类，无需手动操作，并且可以实现内存复用，无需反复申请。
5. 预处理和推理同时进行
6. 生产者消费者模式，合理好用。

## 2_faster_tensorrt

原始的TensorRT_Pro确实有十分优秀的性能，并且接口的设计也十分简单，但是我在复现和使用的时候发现几点可能不太适用于机器人使用的特点。
1. 它的加速是在将需要推理的所有图像全部commit, 然后它内部每个batch的加载和推理, 加速主要集中在预处理和推理同时进行，也就是prefetch的思想。但是在单目机器人上往往是视频流输入，此时是一般是不能输入batch数据的，这个时候实际上是不会比直接推理快多少的。

2. 它用的是自写的CUDA NMS，但是实际上TensorRT8上有一些官方的NMS插件，可以替换。两者的实际效果带测试。


我们应该思考在实际机器人的视频流推理上需要的是什么？

**吞吐和延迟！**

我们的目标肯定是吞吐量超级大同时延迟超级小，在一定条件下，其实这两个是有互斥的意思的。但是另一方面，其实这也是程序两种运行方式，吞吐量代表并行效果，延迟代表串行效果。

1. 为了提高吞吐量，往往会先缓存一些数据，成为一个batch进行推理。也就是相同时间内，可以实现更多图像的推理。BUT，这对于每一帧图像来说，它从输入到输出的延迟就会提高了！

https://github.com/PaddlePaddle/FastDeploy/blob/develop/tutorials/multi_thread/README_CN.md

一个tensorrt的engine可以创建多个context，实现多线程调用。只占用一个engine显存的大小，同时供多个推理运算

NVIDIA Nsight Systems version 2019.5.2.16-b54ef97