# Faster tensorrt


## 文件说明
我在大多数地方都已经加了注释了，应该能够容易看懂。当然注释可能也会有冷不丁写错或者理解错误啥的，还是需要有自己的思考的。
### include
1. `tools.hpp`: 一些工具函数 包括log日志打印，CUDA检查，输出文件保存读取等定义并直接实现
2. `memory_tensor.hpp`: 定义`MixMemory`主要是内存和现存的申请和释放啥的；定义`Tensor`主要是张量的管理和拷贝啥的
3. `monopoly_accocator.hpp`: 内存独占管理分配器 最终实现预处理和推理并行的重要工具
4. `infer_base.hpp`: 包括trt引擎管理类和异步安全推理类
5. `yolo.hpp`: 实现yolo的推理
