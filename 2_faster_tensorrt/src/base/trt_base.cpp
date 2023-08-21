#include "trt_base.hpp"

namespace FasterTRT {

// 返回mode的名字
const char* mode_string(Mode type) {
    switch(type) {
        case Mode::FP32:
            return "FP32";
        case Mode::FP16:
            return "FP16";
        case Mode::INT8:
            return "INT8";
        default:
            return "UnknowCompileMode";
    }
}

//////////////////////////////////////////////////////////////////////
////////////////////// Int8EntropyCalibrator /////////////////////////
//////////////////////////////////////////////////////////////////////
Int8EntropyCalibrator::Int8EntropyCalibrator(const std::vector<std::string>& imagefiles,
                                             nvinfer1::Dims dims, const Int8Process& preprocess) {
    Assert(preprocess != nullptr);
    this->dims_ = dims;
    this->allimgs_ = imagefiles;
    this->preprocess_ = preprocess;
    this->fromCalibratorData_ = false;
    files_.resize(dims.d[0]);
    checkCudaRuntime(cudaStreamCreate(&stream_));
}

Int8EntropyCalibrator::Int8EntropyCalibrator(const std::vector<uint8_t>& entropyCalibratorData,
                                             nvinfer1::Dims dims, const Int8Process& preprocess) {
    Assert(preprocess != nullptr);

    this->dims_ = dims;
    this->entropyCalibratorData_ = entropyCalibratorData;
    this->preprocess_ = preprocess;
    this->fromCalibratorData_ = true;
    files_.resize(dims.d[0]);
    checkCudaRuntime(cudaStreamCreate(&stream_));
}

Int8EntropyCalibrator::~Int8EntropyCalibrator() { checkCudaRuntime(cudaStreamDestroy(stream_)); }

int Int8EntropyCalibrator::getBatchSize() const noexcept { return dims_.d[0]; }

bool Int8EntropyCalibrator::next() {
    int batch_size = dims_.d[0];
    if(cursor_ + batch_size > allimgs_.size()) return false;

    int old_cursor = cursor_;
    for(int i = 0; i < batch_size; ++i) files_[i] = allimgs_[cursor_++];

    if(!tensor_) {
        tensor_.reset(new Tensor(dims_.nbDims, dims_.d));
        tensor_->set_stream(stream_);
        tensor_->set_workspace(std::make_shared<MixMemory>());
    }

    preprocess_(old_cursor, allimgs_.size(), files_, tensor_);
    return true;
}

bool Int8EntropyCalibrator::getBatch(void* bindings[], const char* names[],
                                     int nbBindings) noexcept {
    if(!next()) return false;
    bindings[0] = tensor_->gpu();
    return true;
}

const std::vector<uint8_t>& Int8EntropyCalibrator::getEntropyCalibratorData() {
    return entropyCalibratorData_;
}

const void* Int8EntropyCalibrator::readCalibrationCache(size_t& length) noexcept {
    if(fromCalibratorData_) {
        length = this->entropyCalibratorData_.size();
        return this->entropyCalibratorData_.data();
    }

    length = 0;
    return nullptr;
}

void Int8EntropyCalibrator::writeCalibrationCache(const void* cache, size_t length) noexcept {
    entropyCalibratorData_.assign((uint8_t*)cache, (uint8_t*)cache + length);
}

std::vector<std::string> glob_image_files(const std::string& directory) {
    /* 检索目录下的所有图像："*.jpg;*.png;*.bmp;*.jpeg;*.tiff" */
    std::vector<std::string> files, output;
    std::set<std::string> pattern_set{"jpg", "png", "bmp", "jpeg", "tiff"};

    if(directory.empty()) {
        INFOE("Glob images from folder failed, folder is empty");
        return output;
    }

    try {
        std::vector<cv::String> files_;
        files_.reserve(10000);
        cv::glob(directory + "/*", files_, true);
        files.insert(files.end(), files_.begin(), files_.end());
    } catch(...) {
        INFOE("Glob %s failed", directory.c_str());
        return output;
    }

    for(int i = 0; i < files.size(); ++i) {
        auto& file = files[i];
        int p = file.rfind(".");
        if(p == -1) continue;

        auto suffix = file.substr(p + 1);
        std::transform(suffix.begin(), suffix.end(), suffix.begin(), [](char c) {
            if(c >= 'A' && c <= 'Z') c -= 'A' + 'a';
            return c;
        });
        if(pattern_set.find(suffix) != pattern_set.end()) output.push_back(file);
    }
    return output;
}

}  // namespace FasterTRT