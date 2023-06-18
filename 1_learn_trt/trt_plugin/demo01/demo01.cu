#include "demo01.h"

__global__ void addScalarKernel(const float *input, float *output, const float scalar, const int nElement)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nElement)
        return;

    float _1      = input[index];
    float _2      = _1 + scalar;
    output[index] = _2;
}

namespace nvinfer1
{
ZZX_ADDScalar::ZZX_ADDScalar(const std::string &name, float scalar): name_(name)
{
    WHERE_AM_I();   // debug用的
    m_.scalar = scalar;
}

ZZX_ADDScalar::ZZX_ADDScalar(const std::string &name, const void *buffer, size_t length): name_(name)
{
    WHERE_AM_I();
    memcpy(&m_, buffer, sizeof(m_));
}

ZZX_ADDScalar::~ZZX_ADDScalar()
{
    WHERE_AM_I();
}

// 深拷贝
IPluginV2DynamicExt *ZZX_ADDScalar::clone() const noexcept
{
    WHERE_AM_I();
    auto p = new ZZX_ADDScalar(name_, &m_, sizeof(m_));
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

// 获得输出数量 自定义为1
int32_t ZZX_ADDScalar::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 1;
}

// 获得输出数据类型 自定义为和输入一样
DataType ZZX_ADDScalar::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I();
    return inputTypes[0];
}

// 获取输出维度
DimsExprs ZZX_ADDScalar::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I();
    return inputs[0];
}


bool ZZX_ADDScalar::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    bool res;
    switch (pos)
    {
    case 0:
        res = inOut[0].type == DataType::kFLOAT && inOut[0].format == TensorFormat::kLINEAR;
        break;
    case 1:
        res = inOut[1].type == inOut[0].type && inOut[1].format == inOut[0].format;
        break;
    default: // should NOT be here!
        res = false;
    }
#ifdef DEBUG
    std::cout << "\tpos=" << pos << ",res=" << res << "->[";
    for (int i = 0; i < nbInputs + nbOutputs; ++i)
    {
        std::cout << formatToString(inOut[i].format) << ",";
    }
    std::cout << "],[";
    for (int i = 0; i < nbInputs + nbOutputs; ++i)
    {
        std::cout << dataTypeToString(inOut[i].type) << ",";
    }
    std::cout << "]" << std::endl;
#endif
    return res;
}

// 推理前调用
void ZZX_ADDScalar::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    return;
}

// 告诉trt需要多大中间变量储存空间， 便于后续优化
size_t ZZX_ADDScalar::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 0;
}

// 核心，调用核函数 不要在这里使用cudaMalloc*等函数（导致巨大的申请开销）
int32_t ZZX_ADDScalar::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    int nElement = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
    {
        nElement *= inputDesc[0].dims.d[i];
    }
    dim3 grid(CEIL_DIVIDE(nElement, 256), 1, 1), block(256, 1, 1);
    addScalarKernel<<<grid, block, 0, stream>>>(reinterpret_cast<const float *>(inputs[0]), reinterpret_cast<float *>(outputs[0]), m_.scalar, nElement);
    return 0;
}

// context/engine 销毁的时候调用
void ZZX_ADDScalar::destroy() noexcept
{
    WHERE_AM_I();
    delete this;
    return;
}

// engine 创建时被调用，用于初始化 Plugin
int32_t ZZX_ADDScalar::initialize() noexcept
{
    WHERE_AM_I();
    return 0;
}

// terminate (engine 销毁时被调用，用于释放 initialize 函数申请的资源
void ZZX_ADDScalar::terminate() noexcept
{
    WHERE_AM_I();
    return;
}

// 序列化
// （报告序列化需要的空间大小，单位 Byte
size_t ZZX_ADDScalar::getSerializationSize() const noexcept
{
    WHERE_AM_I();
    return sizeof(m_);
}

// （将 Plugin 数据序列化到给定的 buffer 中）
void ZZX_ADDScalar::serialize(void *buffer) const noexcept
{
    WHERE_AM_I();
    memcpy(buffer, &m_, sizeof(m_));
    return;
}

void ZZX_ADDScalar::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *ZZX_ADDScalar::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *ZZX_ADDScalar::getPluginType() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *ZZX_ADDScalar::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

// （申请使用 context 独占的 cudnn 或 cublas
void ZZX_ADDScalar::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept
{
    WHERE_AM_I();
    return;
}

//（销毁 context 独占的 cudnn 或 cublas 资
void ZZX_ADDScalar::detachFromContext() noexcept
{
    WHERE_AM_I();
    return;
}




// class AddScalarPluginCreator
PluginFieldCollection    ZZXAddScalarPluginCreator::fc_ {};
std::vector<PluginField> ZZXAddScalarPluginCreator::attr_;

ZZXAddScalarPluginCreator::ZZXAddScalarPluginCreator()
{
    WHERE_AM_I();
    attr_.clear();
    attr_.emplace_back(PluginField("scalar", nullptr, PluginFieldType::kFLOAT32, 1));
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

ZZXAddScalarPluginCreator::~ZZXAddScalarPluginCreator()
{
    WHERE_AM_I();
}

// 接受权重，构造这个算子 
IPluginV2DynamicExt* ZZXAddScalarPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
{
    WHERE_AM_I();
    float                          scalar = 0;
    std::map<std::string, float *> parameterMap {{"scalar", &scalar}};

    for (int i = 0; i < fc->nbFields; ++i)
    {
        if (parameterMap.find(fc->fields[i].name) != parameterMap.end())
        {
            *parameterMap[fc->fields[i].name] = *reinterpret_cast<const float *>(fc->fields[i].data);
        }
    }
    ZZX_ADDScalar *pObj = new ZZX_ADDScalar(name, scalar);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

// 反序列化
IPluginV2DynamicExt *ZZXAddScalarPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
{
    WHERE_AM_I();
    ZZX_ADDScalar *pObj = new ZZX_ADDScalar(name, serialData, serialLength);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

void ZZXAddScalarPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *ZZXAddScalarPluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *ZZXAddScalarPluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *ZZXAddScalarPluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

const PluginFieldCollection *ZZXAddScalarPluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(ZZXAddScalarPluginCreator);

}
