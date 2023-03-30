import ctypes
from cuda import cudart
import numpy as np
import os
import tensorrt as trt

SOFILE = './build/libdemo01.so'
np.set_printoptions(precision=3, linewidth=100, suppress=True)  # 控制Python中小数的显示精度
np.random.seed(123456)
cudart.cudaDeviceSynchronize()

def printArrayInfomation(x, info="", n=5):
    print(f"{info}: {x.shape}, SumAbs={np.sum(abs(x)) :.5e}, Var={np.var(x) :.5f}, \
          Max={np.max(x) :.5f},Min={np.min(x) :.5f},SAD={np.sum(np.abs(np.diff(x.reshape(-1)))) :.5f}")
    print('\t', x.reshape(-1)[:n], x.reshape(-1)[-n:])

def check(a, b, weak=False, checkEpsilon=1e-5):
    if weak:
        res = np.all(np.abs(a - b) < checkEpsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + checkEpsilon))
    print(f"check:{res}, absDiff={diff0}, relDiff={diff1}")

def addScalarCPU(inputH, scalar):
    return [inputH[0] + scalar]

def getAddScalarPlugin(scalar):
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == "ZZX_ADDScalar":
            parameterList = []
            parameterList.append(trt.PluginField("scalar", np.float32(scalar), trt.PluginFieldType.FLOAT32))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None


def run(shape, scalar):
    testCase = f"<shape={shape},scalar={scalar}>"
    trtFile = f"./model-Dim{len(shape)}.plan"
    print(f"Test {testCase}")
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(SOFILE)

    if os.path.isfile(trtFile):
        with open(trtFile, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine == None:
            print("Failed loading engine!")
            return
        print("Succeeded loading engine!")
    else:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()

        inputT0 = network.add_input("inputT0", trt.float32, [-1 for i in shape])
        profile.set_shape(inputT0.name, [1 for i in shape], [8 for i in shape], [32 for i in shape])
        config.add_optimization_profile(profile)

        pluginLayer = network.add_plugin_v2([inputT0], getAddScalarPlugin(scalar))
        network.mark_output(pluginLayer.get_output(0))
        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
        with open(trtFile, "wb") as f:
            f.write(engineString)
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    context = engine.create_execution_context()
    context.set_input_shape(lTensorName[0], shape)
    #for i in range(nIO):
    #    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

    bufferH = []
    bufferH.append(np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
    for i in range(nInput, nIO):
        bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
    bufferD = []
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    for i in range(nIO):
        context.set_tensor_address(lTensorName[i], int(bufferD[i]))

    context.execute_async_v3(0)

    for i in range(nInput, nIO):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    outputCPU = addScalarCPU(bufferH[:nInput], scalar)
    """
    for i in range(nInput):
        printArrayInfomation(bufferH[i])
    for i in range(nInput, nIO):
        printArrayInfomation(bufferH[i])
    for i in range(nInput, nIO):
        printArrayInfomation(outputCPU[i - nInput])
    """
    check(bufferH[nInput:][0], outputCPU[0], True)

    for b in bufferD:
        cudart.cudaFree(b)
    print(f"Test {testCase} finish!")

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")

    run([32], 1)
    run([32, 32], 1)
    run([16, 16, 16], 1)
    run([8, 8, 8, 8], 1)
    run([32], 1)
    run([32, 32], 1)
    run([16, 16, 16], 1)
    run([8, 8, 8, 8], 1)

    print("Test all finish!")
