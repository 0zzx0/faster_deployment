#include <iostream>
#include <fstream>
#include <vector>
#include"NvInfer.h"
#include "NvOnnxParser.h"
#include "cookbookHelper.cuh"

using namespace nvinfer1;
// using namespace nvonnxparser;

// const std::string trtfile {"../files/model_cpp.engine"};
// const std::string onnxfile {"../files/model.onnx"};
const char* trtfile="../../files/model_cpp.engine";
const char* onnxfile="../../files/model.onnx";


// logger 
class MyLogger : public ILogger           
{
    void log(Severity severity, const char* msg) noexcept override	//noexcept不会抛出异常。override虚函数重写
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;


void get_engine(){
    IBuilder* builder = createInferBuilder(logger);
    INetworkDefinition* network = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    IBuilderConfig* config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 30);
    /* 这个是指定输入尺寸和输入名字的
        ITensor *inputTensor = network->addInput("inputT0", DataType::kFLOAT, Dims32 {3, {-1, -1, -1}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims32 {3, {1, 1, 1}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims32 {3, {3, 4, 5}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims32 {3, {6, 8, 10}});
        config->addOptimizationProfile(profile);
    */
    // onnx解析器
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    parser->parseFromFile(onnxfile, static_cast<int32_t>(ILogger::Severity::kWARNING));
    for(int32_t i = 0; i < parser->getNbErrors(); ++i){
        std::cout << parser->getError(i)->desc() << std::endl;
    }

    IHostMemory* engineString = builder->buildSerializedNetwork(*network, *config); //创建engine
    if(engineString == nullptr || engineString->size()==0){
        std::cout<<"building 序列化 engine失败"<<std::endl;
        return;
    }

    std::ofstream engineFile(trtfile, std::ios::binary);
    engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
    if (engineFile.fail())
        {
            std::cout << "保存失败" << std::endl;
            return;
        }
    std::cout << "生成成功！" << std::endl;
}



void infer(){
    ICudaEngine *engine = nullptr;

    std::ifstream engineFile(trtfile, std::ios::binary);
    long int fsize = 0;

    engineFile.seekg(0, engineFile.end);    // 指针设到文件最后，也就是继续写入
    fsize = engineFile.tellg();             // 返回当前定位指针的位置，也代表着输入流的大小
    engineFile.seekg(0, engineFile.beg);    // 文件开头
    std::vector<char> engineString(fsize);
    engineFile.read(engineString.data(), fsize);   

    if(engineString.size() == 0){
        std::cout<<"读取序列化数据失败"<<std::endl;
        return;
    } 

    IRuntime* runtime {createInferRuntime(logger)};
    engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
    if(engine == nullptr){
        std::cout<<"反序列化失败"<<std::endl;
        return;
    }

    int nIO = engine->getNbIOTensors(); // io数量
    int nInput = 0;
    int nOutput = 0;
    std::vector<std::string> vTensorName(nIO);
    for (int i = 0; i < nIO; ++i){
        vTensorName[i] = std::string(engine->getIOTensorName(i));
        nInput += int(engine->getTensorIOMode(vTensorName[i].c_str()) == TensorIOMode::kINPUT);
        nOutput += int(engine->getTensorIOMode(vTensorName[i].c_str()) == TensorIOMode::kOUTPUT);
    }

    IExecutionContext* context = engine->createExecutionContext();
    // context->setInputShape(vTensorName[0].c_str(), Dims32 {3, 3, 4, 5});

    // 打印输出输出形状啥的
    for (int i = 0; i < nIO; ++i){
        std::cout<<std::string(i < nInput ? "Input [" : "Output[");
        std::cout<<i << std::string("] -> ");
        std::cout << dataTypeToString(engine->getTensorDataType(vTensorName[i].c_str())) << std::string(" ");
        std::cout << shapeToString(engine->getTensorShape(vTensorName[i].c_str())) << std::string(" ");
        std::cout << shapeToString(context->getTensorShape(vTensorName[i].c_str())) << std::string(" ");
        std::cout << vTensorName[i] << std::endl;
    }
    
    std::vector<int> vTensorSize(nIO, 0);
    for (int i = 0; i < nIO; ++i){
        Dims32 dim  = context->getTensorShape(vTensorName[i].c_str());
        int size = 1;
        for (int j = 0; j < dim.nbDims; ++j){
            size *= dim.d[j];
        }
        vTensorSize[i] = size * dataTypeToSize(engine->getTensorDataType(vTensorName[i].c_str()));
    }

    std::vector<void *> vBufferH {nIO, nullptr};    // cpu
    std::vector<void *> vBufferD {nIO, nullptr};    // gpu
    // gpu分配内存
    for (int i = 0; i < nIO; ++i){
        vBufferH[i] = (void *)new char[vTensorSize[i]];
        CHECK(cudaMalloc(&vBufferD[i], vTensorSize[i]));
    }

    // 赋值
    float *pData = (float *)vBufferH[0];
    for (int i = 0; i < vTensorSize[0] / dataTypeToSize(engine->getTensorDataType(vTensorName[0].c_str())); ++i){
        pData[i] = float(i);
    }

    // 数据复制 cpu -> gpu
    for (int i = 0; i < nInput; ++i){
        CHECK(cudaMemcpy(vBufferD[i], vBufferH[i], vTensorSize[i], cudaMemcpyHostToDevice));
    }

    // gpu上名字对应地址
    for (int i = 0; i < nIO; ++i){
        context->setTensorAddress(vTensorName[i].c_str(), vBufferD[i]);
    }

    // 推理
    context->enqueueV3(0);

    // 数据复制 gpu -> cpu
    for (int i = nInput; i < nIO; ++i){
        CHECK(cudaMemcpy(vBufferH[i], vBufferD[i], vTensorSize[i], cudaMemcpyDeviceToHost));
    }

    // 打印输出
    // for (int i = 0; i < nIO; ++i){
    //     printArrayInfomation((float *)vBufferH[i], context->getTensorShape(vTensorName[i].c_str()), vTensorName[i], true);
    // }

    // 释放内存 释放gpu显存
    for (int i = 0; i < nIO; ++i){
        delete[] vBufferH[i];
        CHECK(cudaFree(vBufferD[i]));
    }

    return;
}


int main(){
    CHECK(cudaSetDevice(0));
    // get_engine();
    infer();
    return 0;
}