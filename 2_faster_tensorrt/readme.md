# Faster tensorrt

## 前言

使用之前你应该已经了解trt的构建和推理流程，所以此处不再涉及基础使用。你应该修改的最少有
```
1. CMakeLists.txt中的cuda、cudnn、trnsorrt环境路径
2. main.cpp中的测试推理图片/视频的路径、trt二进制文件路径，推理类别等
3. 预处理和后处理也要根据实际使用模型修改，本文代码以yolox为例
```

原始的TensorRT_Pro有十分优秀的性能，并且接口的设计也很巧妙。但是我在复现和使用的时候发现部分可能不太适用于我当前使用的机器人。
1. 它的加速是在将需要推理的所有图像全部commit, 然后它内部每个batch的加载和推理, 加速主要集中在batch数据的预处理和推理异步进行，也就是prefetch的思想。但是在单目机器人上往往是视频流输入，此时是一般是不能输入batch数据的，所以此时实际上是不会比直接推理快多少。

2. 但图像commit没有任务队列管理

3. 它用的是自写的CUDA NMS，但是实际上TensorRT8上有一些官方的NMS插件，可以替换。两者的实际效果带测试。



## 1. 文件说明

我在大多数地方都已经加了中文注释，应该能够容易看懂。当然注释可能也会有写错或者理解错误啥的，还是需要有自己的思考的，也欢迎一起交流。

### 1.1 include

1. `tools.hpp`: 一些工具函数 包括log日志打印，CUDA检查，输出文件保存读取等定义并直接实现
2. `memory_tensor.hpp`: 定义`MixMemory`实现内存和显存的申请和释放；定义`Tensor`实现张量的管理、扩容、拷贝等
3. `monopoly_accocator.hpp`: 定义内存独占管理分配器，最终实现预处理和推理并行的重要工具
4. `infer_base.hpp`: 定义trt引擎管理类和异步安全推理类
5. `yolo.hpp`: 定义yolo的推理

### 1.2 src

1. `cuda_kernel.cu`: cuda核函数的实现，预处理和后处理相关的cuda加速代码
2. `memory_tensor.cpp`: `MixMemory`和`Tensor`的实现
3. `infer_base.cpp`: trt引擎管理类和异步安全推理类的实现
4. `yolo.cpp`: yolo推理的实现
5. `main.cpp`: 主函数

### 1.3 src/eval

这是一个评估代码，可以测试相关数据集(coco格式)使用trt推理的map。

1. `save.hpp`: 一个保存检测结果到文件里的类
2. `get_imgid_txt.py`: 读取`eval_results.json`，来保存图片name和id到文件`img_id.txt`
3. `eval.cpp`: 读取`img_id.txt`中的图片，进行推理，并保存相应结果到`results.txt`
4. `img_id.txt`: img的id和img的name的对应，便于评估
5. `results.txt`: 检测的结果
6. `eval_results.json`: 待评估的coco类型的文件
7. `eval.py`: 最终的评估程序，打印结果


## 2. 使用教程

### 2.1 模型转换



#### 2.1.1 trtexec
模型转换部分，在不需要增加自定义算子的时候，想要导出tensorrt的engine，**trtexec is all you need！**

```shell
# 构建模型的时候
trtexec 
    --onnx = ./model NCHW.onnx  # 指定onnx模型文件名
    # --output=y:0                # 指定输出张量名（使用 Onnx 时该选项无效）
    --minShapes =x:0:1x1x28x28
    --optShapes =x:0:4x1x28x28
    --maxShapes =x:0:16x1x28x28 # 指定输入形状的范围最小值、最常见值、最大值
    --workspace = 1024   # 以后要用 memPoolSize 优化过程可使用显存最大值
    --fp16  	         # 指定引擎精度和稀疏性等属性 int8 noTF32 best sparsity
    --saveEngine=model.plan # 指定输出引擎文件名
    --skipInference         # 只创建引擎不运行 旧版本叫buildonly
    --verbose 	            # 打印详细日志
    --timingCacheFile=timing.cache # 指定输出优化计时缓存文件名
    --profilingVerbosity =detailed # 构建期保留更多的逐层信息
    --dumpLayerInfo                 # 打印层信息
    --exportLayerInfo=layerInfo.txt # 导出引擎逐层信息，可与 profilingVerbosity 合用


# 运行的时候
trtexec 
    --loadEngine=model.plan # 读取 engine 文件
    --shapes=x:1x1x28x28    # 指定输入张量形状
    --warmUp=1000           # 热身阶段最短运行时间（单位： ms
    --duration=10           # 测试阶段最短运行时间（单位： s
    --iterations=100        # 指定测试阶段运行的最小迭代次数
    --useCudaGraph          # 使用 CUDAGraph 来捕获和执行推理过程 
    --noDataTransfers       # 关闭 Host 和 Device 之间的数据传输
    --streams=2             # 使用多个 stream 来运行推理
    --verbose               # 打印详细日志
    --dumpProfile 
    --exportProfile=layerProfile.txt 	# 保存逐层性能数据信息

```


#### 2.1.2 polygraphy
很牛的工具！

polygraphy工具，可以多后端运行对比，对比不同后端结果，生成engine等（重要），还可以判断那些算子不能被trt加速，并把这些切割出来
Build TensorRT engine using the ONNX file, and compare the output of each layer between Onnxruntime and TensorRT
```shell
polygraphy run model.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --save-engine=model-FP32-MarkAll.plan \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --onnx-outputs mark all \
    --trt-outputs mark all \
    --trt-min-shapes 'tensor-0:[1,1,28,28]' \
    --trt-opt-shapes 'tensor-0:[4,1,28,28]' \
    --trt-max-shapes 'tensor-0:[16,1,28,28]' \
    --input-shapes   'tensor-0:[4,1,28,28]'
    > result-run-FP32-MarkAll.log 2>&1

```

#### 2.1.3 trt api
除此之外，tensorrt_pro中也给出了一个complie的模型转换接口，我也搬运了过来
```cpp

bool compile(
    Mode mode,
    YoloType type,
    unsigned int max_batch_size,
    const string& source_onnx_file,
    const string& save_engine_file,
    size_t max_workspace_size = 1<<30,
    const string& int8_images_folder="",
    const string& int8_entropy_calibrator_cache_file=""
);

```


### 2.2 模型推理

这个项目的一个优点就是接口简单，尤其是推理接口。
```cpp
// 创建模型
auto yolo = YOLO::create_infer(model_file, type, deviceid, batch_size, confidence_threshold, nms_threshold);

// 推理图片
auto objs = yolo->commit(image);

// 得到结果
auto res = objs.get();

```


### 2.3 模型测评

使用c++的推理结果来实现coco格式的eval格式，进而便于对比加速前后精度的变化。稍微有点麻烦，整体思想是保存c++的推理结果，然后用python的pycocotools来实现结果的计算。

首先运行`eval/get_imgid_txt.py`，得到`img_id.txt`文件，包含了图片名称和图片id的对应
```
0 005894.jpg
1 004755.jpg
```

然后默认cmake会编译eval文件夹的内容，当需要模型评测时，运行`build/eval`可以得到`results.txt`,包含推理结果
```
005894.jpg 0 0 0.836939 1175 609 229 181 
005894.jpg 0 1 0.768631 2468 1880 99 162 
005894.jpg 0 2 0.70347 1938 607 216 141 
005894.jpg 0 2 0.781555 944 1442 163 203 
004755.jpg 1 1 0.557236 622 361 59 45 
004755.jpg 1 1 0.676005 383 79 64 44 
```
最后运行`eval/eval.py`，得到最终的coco格式的map
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.447
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.751
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.486
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.175
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.404
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.503
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.166
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.465
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.507
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.244
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.471
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.558
```


### 2.4 自定义模型



## 3. More优化

<!-- - [ ] 预分配内存池，避免在推理阶段重复分配 -->
- [ ] 多流
- [ ] gpu内存异步操作内核融合，使用一个gpu内核实现运算符组合，减少数据传输和内核启动延迟
- [ ] 一个tensorrt的engine可以创建多个context，实现多线程调用。只占用一个engine显存的大小，同时供多个推理运算
<!-- - [ ] 拷贝的异步 -->
<!-- - [ ] 向量化全局内存访问，提高内存访问效率 -->
