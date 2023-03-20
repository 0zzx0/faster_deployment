import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from cuda import cudart

model_path = '../files/model.onnx'
model_engine_path = '../files/model_py.engine'


def get_engine():
    # 构建阶段
    logger = trt.Logger(trt.Logger.WARNING)     # logger
    builder = trt.Builder(logger)               # builder

    # 创建网络定义
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()     # 动态尺寸的话需要这个 
    config = builder.create_builder_config()            # 配置
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20) # 1 MiB

    # 创建解析器
    parser = trt.OnnxParser(network, logger)
    success = parser.parse_from_file(model_path)    # 加载文件
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    # 构建engine
    serialized_engine = builder.build_serialized_network(network, config)

    with open(model_engine_path, "wb") as f:
        f.write(serialized_engine)

def inferV1():
    logger = trt.Logger(trt.Logger.WARNING)     # logger

    runtime = trt.Runtime(logger)
    with open(model_engine_path, "rb") as f:
        serialized_engine = f.read()
    engine = runtime.deserialize_cuda_engine(serialized_engine)

    
    stream = cuda.Stream()
    context = engine.create_execution_context()

    h_input = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    with engine.create_execution_context() as context:
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input, h_input, stream)
        # Run inference.
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        # Return the host output. 该数据等同于原始模型的输出数据
    return h_output


def inferV2():
    logger = trt.Logger(trt.Logger.WARNING)     # logger

    runtime = trt.Runtime(logger)
    with open(model_engine_path, "rb") as f:
        serialized_engine = f.read()
    engine = runtime.deserialize_cuda_engine(serialized_engine)

    nIO = engine.num_io_tensors     # io变量数量
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]   # 获取io变量名字
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT) # 输入tensor数量
    Output = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.OUTPUT)

    context = engine.create_execution_context()
    print("===============INPUT/OUTPUT=================== ")
    for i in range(nIO):
        print(f"[{i}]{'Input ' if i < nInput else 'Output'} -> "+
              f"{engine.get_tensor_dtype(lTensorName[i])} " +       # 数据类型
              f"{engine.get_tensor_shape(lTensorName[i])} " +       # engine形状
              f"{context.get_tensor_shape(lTensorName[i])} " +      # context形状
              f"{lTensorName[i]} ")                                 # 名字
    print("============================================ ")
    
    data = np.arange(3 * 224 * 224, dtype=np.float32).reshape(1, 3, 224, 224)   

    # cpu端数据
    bufferH = []                                                            
    bufferH.append(np.ascontiguousarray(data))  # 输入数据转内存连续
    for i in range(nInput, nIO):                # 输出数据
        bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
    
    # gpu端数据申请显存
    bufferD = []
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])
    
    # 输入数据复制到显存
    for i in range(nInput):                                                    
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    # 设置输入输出数据的地址(buffer)
    for i in range(nIO):
        context.set_tensor_address(lTensorName[i], int(bufferD[i]))            

    # 推理
    context.execute_async_v3(0) 

    for i in range(nInput, nIO):    # 数据拷会cpu
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for i in range(nIO):
        print(f'{lTensorName[i]}:\t {bufferH[i].shape}')


    for b in bufferD:       # 释放显存  
        cudart.cudaFree(b)


if __name__ == '__main__':
    # get_engine()
    # print(infer().shape)
    inferV2()   # 推荐
