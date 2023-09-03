#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"
#include "logging.h"

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0  // GPU id
static const int INPUT_W = 640;
static const int INPUT_H = 640;
const char* INPUT_BLOB_NAME = "images";
const char* OUTPUT_BLOB_NAME1 = "output";
const char* OUTPUT_BLOB_NAME2 = "943";
const char* OUTPUT_BLOB_NAME3 = "944";
const char* OUTPUT_BLOB_NAME4 = "945";

const std::vector<std::string> class_names={"echinus", "starfish", "holothurian", "scallop"};
const std::vector<std::vector<float>> color_list =
{
    {0.000, 0.447, 0.741},
    {0.850, 0.325, 0.098},
    {0.929, 0.694, 0.125},
    {0.494, 0.184, 0.556}
};

using namespace nvinfer1;

// class Logger : public nvinfer1::ILogger {
// public:
//     void log(Severity severity, const char* msg) noexcept override {
//         if (severity != Severity::kINFO) {
//             std::cout << msg << std::endl;
//         }
//     }
// };


class YoloEnd2End{
public:
    YoloEnd2End(const std::string model_path);
    cv::Mat static_resize(cv::Mat& image);
    float* blobFromImage(cv::Mat& img);
    void draw_objects(const cv::Mat& img, float* Boxes, int* ClassIndexs, int* BboxNum);
    void Infer(cv::Mat& img, float* Boxes, float* score, int* ClassIndexs, int* BboxNum);
    ~YoloEnd2End();

private:
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t stream = nullptr;
    void* buffs[5];
    int iH, iW, in_size, out_size1, out_size2, out_size3, out_size4;
    Logger gLogger;
};

// resize
cv::Mat YoloEnd2End::static_resize(cv::Mat& img) {
    float r = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
    // r = std::min(r, 1.0f);
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

// float* YoloEnd2End::blobFromImage(cv::Mat& img){
//     float* blob = new float[img.total()*3];
//     // std::memcpy(blob, img.data, img.total() * 3* sizeof(float));
//     int img_h = img.rows;
//     int img_w = img.cols;
//     int channelLength = img_w * img_h;
//     std::vector<cv::Mat> split_img = {
//                 cv::Mat(img_h, img_w, CV_32FC1, blob + channelLength * 0),
//                 cv::Mat(img_h, img_w, CV_32FC1, blob + channelLength * 1),
//                 cv::Mat(img_h, img_w, CV_32FC1, blob + channelLength * 2)
//         };
//     cv::split(img, split_img);
//     return blob;
// }


float* YoloEnd2End::blobFromImage(cv::Mat& img){
    float* blob = new float[img.total()*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                blob[c * img_w * img_h + h * img_w + w] =
                    (float)img.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
    return blob;
}

YoloEnd2End::YoloEnd2End(const std::string model_path) {
    std::ifstream ifile(model_path, std::ios::in | std::ios::binary);
    if (!ifile) {
        std::cout << "read serialized file failed\n";
        std::abort();
    }

    ifile.seekg(0, std::ios::end);
    const int mdsize = ifile.tellg();
    ifile.clear();
    ifile.seekg(0, std::ios::beg);
    std::vector<char> buf(mdsize);
    ifile.read(&buf[0], mdsize);
    ifile.close();
    std::cout << "model size: " << mdsize << std::endl;

    runtime = nvinfer1::createInferRuntime(gLogger);
    initLibNvInferPlugins(&gLogger, "");
    engine = runtime->deserializeCudaEngine((void*)&buf[0], mdsize, nullptr);

    auto in_dims = engine->getTensorShape(INPUT_BLOB_NAME);
    // auto in_dims = engine->getBindingDimensions(engine->getBindingIndex("images"));
    iH = in_dims.d[2];
    iW = in_dims.d[3];
    in_size = 1;
    for (int j = 0; j < in_dims.nbDims; j++) {
        in_size *= in_dims.d[j];
    }
    auto out_dims1 = engine->getTensorShape(OUTPUT_BLOB_NAME1);
    // auto out_dims1 = engine->getBindingDimensions(engine->getBindingIndex("output"));
    out_size1 = 1;
    for (int j = 0; j < out_dims1.nbDims; j++) {
        out_size1 *= out_dims1.d[j];
    }
    auto out_dims2 = engine->getTensorShape(OUTPUT_BLOB_NAME2);
    // auto out_dims2 = engine->getBindingDimensions(engine->getBindingIndex("943"));
    out_size2 = 1;
    for (int j = 0; j < out_dims2.nbDims; j++) {
        out_size2 *= out_dims2.d[j];
    }
    auto out_dims3 = engine->getTensorShape(OUTPUT_BLOB_NAME3);
    // auto out_dims3 = engine->getBindingDimensions(engine->getBindingIndex("944"));
    out_size3 = 1;
    for (int j = 0; j < out_dims3.nbDims; j++) {
        out_size3 *= out_dims3.d[j];
    }
    auto out_dims4 = engine->getTensorShape(OUTPUT_BLOB_NAME4);
    // auto out_dims4 = engine->getBindingDimensions(engine->getBindingIndex("945"));
    out_size4 = 1;
    for (int j = 0; j < out_dims4.nbDims; j++) {
        out_size4 *= out_dims4.d[j];
    }

    context = engine->createExecutionContext();
    if (!context) {
        std::cout << "create execution context failed\n";
        std::abort();
    }

    CHECK(cudaMalloc(&buffs[0], in_size * sizeof(float)));
    CHECK(cudaMalloc(&buffs[1], out_size1 * sizeof(int)));
    CHECK(cudaMalloc(&buffs[2], out_size2 * sizeof(float)));
    CHECK(cudaMalloc(&buffs[3], out_size3 * sizeof(float)));
    CHECK(cudaMalloc(&buffs[4], out_size4 * sizeof(int)));
    CHECK(cudaStreamCreate(&stream));
}

void YoloEnd2End::Infer(cv::Mat& img, float* Boxes, float* score, int* ClassIndexs, int* BboxNum) {

    cv::Mat pr_img;
    pr_img = this->static_resize(img);
    float* blob = this->blobFromImage(pr_img);
    float scale = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));

    static int* num_dets = new int[out_size1];
    static float* det_boxes = new float[out_size2];
    static float* det_scores = new float[out_size3];
    static int* det_classes = new int[out_size4];

    CHECK(cudaMemcpyAsync(buffs[0], &blob[0], in_size * sizeof(float), cudaMemcpyHostToDevice, stream));
      
    context->enqueueV2(&buffs[0], stream, nullptr);

    CHECK(cudaMemcpyAsync(num_dets, buffs[1], out_size1 * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(det_boxes, buffs[2], out_size2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(det_scores, buffs[3], out_size3 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(det_classes, buffs[4], out_size4 * sizeof(int), cudaMemcpyDeviceToHost, stream));
  
    BboxNum[0] = num_dets[0];
    int img_w = img.cols;
    int img_h = img.rows;
    for (size_t i = 0; i < num_dets[0]; i++) {
        float x0 = (det_boxes[i * 4]) / scale;
        float y0 = (det_boxes[i * 4 + 1]) / scale;
        float w = (det_boxes[i * 4 + 2]) / scale;
        float h = (det_boxes[i * 4 + 3]) / scale;

        x0 = x0 - w/2.0;
        y0 = y0 - h/2.0;
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        w = std::max(w, 0.f);
        h = std::max(h, 0.f);
        Boxes[i * 4] = x0;
        Boxes[i * 4 + 1] = y0;
        Boxes[i * 4 + 2] = w;
        Boxes[i * 4 + 3] = h;
        ClassIndexs[i] = det_classes[i];
        score[i*4] = det_scores[i];
    }
    delete blob;
}

void YoloEnd2End::draw_objects(const cv::Mat& img, float* Boxes, int* ClassIndexs, int* BboxNum) {
    cv::Mat image = img.clone();
    for (int j = 0; j < BboxNum[0]; j++) {
        cv::Rect rect(Boxes[j * 4], Boxes[j * 4 + 1], Boxes[j * 4 + 2], Boxes[j * 4 + 3]);

        cv::Scalar color = cv::Scalar(color_list[ClassIndexs[j]][0], 
                                        color_list[ClassIndexs[j]][1], 
                                        color_list[ClassIndexs[j]][2]);
  
        cv::rectangle(image, rect, color * 255, 2);
        cv::putText(
            image,
            class_names[ClassIndexs[j]],
            cv::Point(rect.x, rect.y - 1),
            cv::FONT_HERSHEY_PLAIN,
            1.2,
            color * 255,
            2);
        cv::imwrite("result.jpg", image);
    }
}

YoloEnd2End::~YoloEnd2End() {
    std::cout<<"释放内存、显存"<<std::endl;
    cudaStreamSynchronize(stream);
    CHECK(cudaFree(buffs[0]));
    CHECK(cudaFree(buffs[1]));
    CHECK(cudaFree(buffs[2]));
    CHECK(cudaFree(buffs[3]));
    CHECK(cudaFree(buffs[4]));
    cudaStreamDestroy(stream);
    delete context;
    delete engine;
    delete runtime;
    // context->destroy();
    // engine->destroy();
    // runtime->destroy();
}


int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    const std::string input_image_path = "../../../../2_faster_tensorrt/inference/1.jpg";
    const std::string engine_file_path="../../../../2_faster_tensorrt/yolox_end2end.engine";

  
    float* Boxes = new float[400];
    float* Scores = new float[100];
    int* BboxNum = new int[1];
    int* ClassIndexs = new int[100];
    YoloEnd2End yolo_end2end(engine_file_path);
    cv::Mat img;
    img = cv::imread(input_image_path);
    // warmup 
    for (int num =0; num < 500; num++) {
        yolo_end2end.Infer(img, Boxes, Scores, ClassIndexs, BboxNum);
    }
    // inference
    auto start = std::chrono::system_clock::now();
    for (int num = 0; num < 1000; num++) {
        yolo_end2end.Infer(img, Boxes, Scores, ClassIndexs, BboxNum);
    }
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() /1000.0<< "ms" << std::endl;

    // std::cout<<BboxNum[0]<<std::endl;
    // std::cout<<ClassIndexs[0]<<" "<<ClassIndexs[1]<<std::endl;
    // std::cout<<Boxes[0]<<" "<<Boxes[1]<<" "<<Boxes[2]<<" "<<Boxes[3]<<std::endl;

    yolo_end2end.draw_objects(img, Boxes, ClassIndexs, BboxNum);

    delete[] Boxes;
    delete[] Scores;
    delete[] BboxNum;
    delete[] ClassIndexs;

    return 0;

}
