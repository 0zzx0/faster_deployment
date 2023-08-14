#ifndef YOLO_HPP
#define YOLO_HPP

/*
    重写改进，集成高性能yolox推理接口
    date: 2023-6-11
    author: zzx
    refer:  https://github.com/shouxieai/tensorRT_Pro.git
*/

#include <memory>
#include <future>
#include <opencv2/opencv.hpp>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include "../../base/tools.hpp"
#include "../../base/memory_tensor.hpp"
#include "../../base/monopoly_accocator.hpp"
#include "../../base/infer_base.hpp"

namespace YOLO{

using namespace std;
using namespace nvinfer1;

///////////////////////TRT/////////////////////////////
#define TRT_STR(v)  #v
#define TRT_VERSION_STRING(major, minor, patch, build)   TRT_STR(major) "." TRT_STR(minor) "." TRT_STR(patch) "." TRT_STR(build)
const char* trt_version();

//////////////////////模型选择//////////////////////////
enum class YoloType : int{V5 = 0, X  = 1};
// 推理数据类型
enum class Mode : int{FP32, FP16, INT8};

const char* trt_version();
const char* mode_string(Mode type);
const char* type_name(YoloType type);
void set_device(int device_id);

////////////////////量化用的///////////////////////////
typedef std::function<void(int current, int count, const std::vector<std::string>& files, std::shared_ptr<Tensor>& tensor)> Int8Process;


///////////////////////// 推理结果格式////////////////
struct Box{
    float left, top, right, bottom, confidence;
    int class_label;

    Box() = default;
    Box(float left, float top, float right, float bottom, float confidence, int class_label)
    :left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label){}
};
typedef vector<Box> BoxArray;


/* Decode配置的实现
    不同模型输出的decode都不一样，即使是yolo系列也由一些区别，
    这里需要实现不同模型的decode。尤其是anchor base 和anchor free的区别
*/
struct DecodeMeta{
    int num_anchor;
    int num_level;
    float w[16], h[16];
    int strides[16];

    // static DecodeMeta v5_p5_default_meta();
    static DecodeMeta x_default_meta();
};


/*
推理的虚基类 最终暴露给用户的只有这个接口
实际推理的类应该继承并实现下面两个纯虚函数
*/
class Infer{
public:
    virtual shared_future<BoxArray> commit(const cv::Mat& image) = 0;
    virtual vector<shared_future<BoxArray>> commits(const vector<cv::Mat> &images) = 0;
};


// 仿射变换矩阵
// struct AffineMatrix{
//     float i2d[3];       // image to dst(network)
//     float d2i[3];       // dst to image

//     void compute(const cv::Size& from, const cv::Size& to){
//         float scale_x = to.width / (float)from.width;
//         float scale_y = to.height / (float)from.height;
//         float scale = std::min(scale_x, scale_y);
//         i2d[0] = scale;  
//         i2d[1] = -scale * from.width  * 0.5 + to.width  * 0.5 + scale * 0.5 - 0.5;
//         i2d[2] = -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;

//         // y = kx + b
//         // x = (y-b)/k = y*1/k + (-b)/k
//         d2i[0] = 1/scale; 
//         d2i[1] = -i2d[1] / scale;
//         d2i[2] = -i2d[2] / scale;
//     }
// };
struct AffineMatrix{
    float i2d[6];       // image to dst(network), 2x3 matrix
    float d2i[6];       // dst to image, 2x3 matrix

    void compute(const cv::Size& from, const cv::Size& to){
        float scale_x = to.width / (float)from.width;
        float scale_y = to.height / (float)from.height;
        float scale = std::min(scale_x, scale_y);
        i2d[0] = scale;  i2d[1] = 0;  i2d[2] = -scale * from.width  * 0.5  + to.width * 0.5 + scale * 0.5 - 0.5;
        i2d[3] = 0;  i2d[4] = scale;  i2d[5] = -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;

        cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
        cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
        cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
    }

    cv::Mat i2d_mat(){
        return cv::Mat(2, 3, CV_32F, i2d);
    }
};

// 线程安全模板类设置模板类型
using ThreadSafedAsyncInferImpl = ThreadSafedAsyncInfer
<
    cv::Mat,                    // input
    BoxArray,                   // output
    tuple<string, int>,         // start param
    AffineMatrix                // additional
>;

/* Yolo的具体实现
    通过上述类的特性，实现预处理的计算重叠、异步跨线程调用，最终拼接为多个图为一个batch进行推理。
    最大化的利用卡性能，实现高性能高yolo推理
*/
class YoloTRTInferImpl : public Infer, public ThreadSafedAsyncInferImpl{
public:

    // 析构 调用来自基类ThreadSafedAsyncInferImpl的stop函数
    ~YoloTRTInferImpl();

    
    virtual bool startup(const string& file, YoloType type, int gpuid, int batch_size, float confidence_threshold, float nms_threshold);
    virtual void worker(promise<bool>& result) override;
    virtual bool preprocess(Job& job, const Mat& image) override;

    virtual vector<shared_future<BoxArray>> commits(const vector<Mat>& images) override;
    virtual shared_future<BoxArray> commit(const Mat& image) override;

    void init_yolox_prior_box(Tensor& prior_box);
    void init_yolov5_prior_box(Tensor& prior_box);

private:
    int input_width_            = 0;
    int input_height_           = 0;
    int gpu_                    = 0;
    float confidence_threshold_ = 0;
    float nms_threshold_        = 0;
    cudaStream_t stream_       = nullptr;
    cudaStream_t stream_pro_   = nullptr;
    Norm normalize_;
    YoloType type_;
    DecodeMeta meta_;
    int batch_size_ = 1;
};



/* 
int8 量化 未测试
*/
class Int8EntropyCalibrator : public IInt8EntropyCalibrator2{
public:
    Int8EntropyCalibrator(const vector<string>& imagefiles, nvinfer1::Dims dims, const Int8Process& preprocess);
    Int8EntropyCalibrator(const vector<uint8_t>& entropyCalibratorData, nvinfer1::Dims dims, const Int8Process& preprocess);
    virtual ~Int8EntropyCalibrator();

    int getBatchSize() const noexcept;
    bool next();
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept;

    const vector<uint8_t>& getEntropyCalibratorData();
    const void* readCalibrationCache(size_t& length) noexcept;
    virtual void writeCalibrationCache(const void* cache, size_t length) noexcept;

private:
    Int8Process preprocess_;
    vector<string> allimgs_;
    size_t batchCudaSize_ = 0;
    int cursor_ = 0;
    nvinfer1::Dims dims_;
    vector<string> files_;
    shared_ptr<Tensor> tensor_;
    vector<uint8_t> entropyCalibratorData_;
    bool fromCalibratorData_ = false;
    cudaStream_t stream_ = nullptr;
};


/* 
    trt模型编译(不过我实际建议直接用trtexec转换,嘻嘻0_0)
    max max_batch_size：为最大可以允许的batch数量
    source_onnx_file：onnx文件
    save_engine_file：储存的tensorRT模型
    max_workspace_size：最大工作空间大小，一般给1GB，在嵌入式可以改为256MB，单位是byte
    int8 images folder：对于Mode为INT8时，需要提供图像数据进行标定，请提供文件夹，会自动检索下面的jpg/jpeg/tiff/png/bmp
    int8_entropy_calibrator_cache_file：对于int8模式下，熵文件可以缓存，避免二次加载数据，可以跨平台使用，是一个txt文件
*/
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

void image_to_tensor(const cv::Mat& image, shared_ptr<Tensor>& tensor, YoloType type, int ibatch);
vector<string> glob_image_files(const string& directory);


shared_ptr<Infer> create_infer(
    const string& engine_file,
    YoloType type,
    int gpuid,
    int batch_size,
    float confidence_threshold=0.2f,
    float nms_threshold=0.5f
    );


}; // end namespace YOLO

#endif