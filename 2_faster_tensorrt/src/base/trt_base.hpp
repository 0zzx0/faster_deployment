#ifndef TRT_BASE_H
#define TRT_BASE_H


#include <NvOnnxParser.h>

#include "memory_tensor.hpp"
#include "monopoly_accocator.hpp"
#include "infer_base.hpp"

namespace FasterTRT{

// 推理数据类型
enum class Mode : int{FP32, FP16, INT8};
const char* mode_string(Mode type);

////////////////////量化用的///////////////////////////
typedef std::function<void(int current, int count, const std::vector<std::string>& files, std::shared_ptr<Tensor>& tensor)> Int8Process;

/* 
int8 量化 未测试
*/
class Int8EntropyCalibrator : public IInt8EntropyCalibrator2{
public:
    Int8EntropyCalibrator(const std::vector<std::string>& imagefiles, nvinfer1::Dims dims, const Int8Process& preprocess);
    Int8EntropyCalibrator(const std::vector<uint8_t>& entropyCalibratorData, nvinfer1::Dims dims, const Int8Process& preprocess);
    virtual ~Int8EntropyCalibrator();

    int getBatchSize() const noexcept;
    bool next();
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept;

    const std::vector<uint8_t>& getEntropyCalibratorData();
    const void* readCalibrationCache(size_t& length) noexcept;
    virtual void writeCalibrationCache(const void* cache, size_t length) noexcept;

private:
    Int8Process preprocess_;
    std::vector<std::string> allimgs_;
    size_t batchCudaSize_ = 0;
    int cursor_ = 0;
    nvinfer1::Dims dims_;
    std::vector<std::string> files_;
    std::shared_ptr<Tensor> tensor_;
    std::vector<uint8_t> entropyCalibratorData_;
    bool fromCalibratorData_ = false;
    cudaStream_t stream_ = nullptr;
};


// 检索目录下的所有图像："*.jpg;*.png;*.bmp;*.jpeg;*.tiff" 
std::vector<std::string> glob_image_files(const std::string& directory);


}

#endif
