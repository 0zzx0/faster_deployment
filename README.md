# Faster Deployment

> 作者本人能力和想法都十分有限，确实可能很多情况没有想到，欢迎大家讨论！

本仓库主要是针对深度学习模型的TensorRT、ncnn、rknn等的部署工作，有较好的接口便捷性和推理性能，暂时均以目标检测部署为例。另外本项目主要当前主要应用在单目机器人，所以benchmark一般设置`batch=1`，采用一张接着一张图片输入或者单视频流输入的方式进行测试，模拟实际情况。

目前建议优先使用faster_tensorrt，因为这是我主要的提升方向，支持的算法最多，后续更新应该也会更快。另外两个也以可能也会慢慢更新，但是速度稍慢。

- 1_trt_learn: 主要是tensorrt的基础操作，包括模型转换、推理、构建插件以及一些运行和优化的demo
- 2_faster_tensorrt: 主要是tensorrt的封装和优化
- 3_faster_ncnn: 参考`2_faster_tensort`封装ncnn推理过程
- 4_faser_rknn: 封装rknn的推理过程

## 致谢

首先需要感谢手写ai团队开源的[TensorRT_Pro](https://github.com/shouxieai/tensorRT_Pro)，让我受益良多，本仓库中tensorrt的代码也均是在其基础上进行优化，以及按照该仓库的整体思路优化ncnn和rknn的推理。

<!-- 该仓库的部分优点：
1. 接口简单清晰
2. 预处理和后处理自写CUDA加速
3. batch可根据实际数据动态调整(前提是trtmodel转换中设置动态batch)
4. 写了内存和数据的管理类，无需手动操作，并且可以实现内存复用，无需反复申请。
5. 预处理和推理同时进行
6. 生产者消费者模式，合理好用。 -->

## 当前支持

### faster_tensorrt

#### 目标检测

- [x] yolox
- [x] yolov8
- [x] rtdetr

#### 单目深度估计

- [ ] [lite-mono](https://github.com/noahzn/Lite-Mono)
https://zhuanlan.zhihu.com/p/614680720

#### 语义分割

### faster_ncnn

#### 目标检测
- [x] yolox
- [ ] yolov8


### faster_rknn

#### 目标检测
- [x] yolox
- [ ] yolov8


## 问题&分析

我们首先应该思考在实际机器人的视频流推理上需要的是什么？

明确前提：我们此时的模型已经充分优化过了，包括剪枝、量化之类的操作之后或者后端推理器已经对模型算子进行了自动或手动的融合、量化、图优化等操作。<u>*简而言之，可以认为单幅图像的纯inference时间是不可能再缩短了。*</u>

<font size=5 >**🔥高吞吐和低延迟！！！** </font>

> 本文中两个词的意义 :   <br>
> **延迟**：图片从诞生到推理完成需要的时间。 <br>
> **吞吐**：相等时间内处理图片的数量。 <br>
> 发现一个更好的解释，来自trt的文档[性能评估](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#measure-performance)


我们的目标肯定是吞吐量特别大，同时延迟超级小，在一定条件下，其实这两个是有互斥的意思的。但是另一方面，其实这也是表明程序性能两种方式，吞吐量代表并行能力，延迟代表串行效果。
> 之所以说有吞吐和延迟互斥的意思是因为，在服务器测部署时，为了提高吞吐量，往往会先缓存一些数据，成为一个batch进行推理(也是tensorrt_pro的策略)。也就是相同时间内，可以实现更多图像的推理，同时毫无疑问这在并行能力高的GPU设备上是十分有效的，可以极大提升吞吐量(显存和算力足够情况下可以轻松提升10倍以上)。但是，这对于每一帧图像来说，它从输入到输出的延迟就会提高了！这对实际机器人来说是显然无法接受的，因为控制的核心还是负反馈，当信号量频率低或者不稳定时，对于机器人的控制和决策来说难度很大。

值得注意的是，当前情况下单幅图像的延迟，在我们这个层面是无法缩短的哦！因为一幅图像进入模型必须经过`预处理->推理->后处理`三个步骤，这三个步骤的时间耗时在本阶段是无法缩小的(这是在之前优化网络结构和模型转换的时候考虑的)，本项目是希望在不增加单图延迟的基础上，尽量高的提升模型检测吞吐量，也就是尽量重合、去掉一些无用的、重复的、耗时的操作，对于机器人来说就是可以拥有更高频率的目标位置信息等。


另外采用多线程或者线程池推理，对机器人处理图像带来的问题是：在输入图像帧率很高的时候，无法保证图像按照输入顺序输出，这对于机器人这种需要根据目标前后运动状态进行决策的智能体来说是有问题的，可能会导致误判。不过我认为如果可以实现这是一个跨越量级的提高吞吐量的方法，当然前提是不增加延迟并且输出有序，我会在这里继续尝试。

## 实现

具体的实现过程在这里[2_faster_tensorrt readme](./2_faster_tensorrt/readme.md)，有完整的代码解释、模型推理接口、增加模型方法等的说明。


## 总结

那本仓库要做的是什么？

1. 首先我暂时抛弃了多batch，因为我当前使用的机器人不需要多输入。
2. 设置任务队列(超过则阻塞，尽量不增加延迟)，可以根据模型预处理和推理耗时手动调整，保证最优的吞吐和延迟。
3. 尽量在可加速的硬件上执行预处理和后处理。



## other

类似百度的fastdeploy、mmdeploy等部署仓库都是有好有坏。首先它们对自家框架都支持的比较好比如paddledetection, mmlab系列的模型仓库等，但是缺点就是现在后端框架的api的更新可能比较快，有时可能无法用到最新的版本和接口，而且说实话，开源仓库的维护成本确实比较大，所以他们可能更新的会稍微慢一点。不过另一方面不愧是大厂，他们的对于代码仓库整体设计思想确实都是非常好的，可扩展性都贼拉好(相对本仓库)，后期我也会慢慢学习学习，来优化本仓库。

https://github.com/PaddlePaddle/FastDeploy/blob/develop/tutorials/multi_thread/README_CN.md

