#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <thread>

#include "net.h"

namespace NCNN_DET{

// bboxes
class ObjBox{

public:
    float GetWidth() { return (x2 - x1); };
    float GetHeight() { return (y2 - y1); };
    float area() { return GetWidth() * GetHeight(); };

    int x1;
    int y1;
    int x2;
    int y2;

    int category;
    float score;
};

struct GridAndStride{
    int grid0;
    int grid1;
    int stride;
};


float InterSectionArea(const ObjBox &a, const ObjBox &b);
bool ScoreSort(ObjBox a, ObjBox b);
void nms(std::vector<ObjBox> &src_boxes, std::vector<ObjBox> &dst_boxes, float threshold);

class Det
{

public:
    Det(int input_w, int input_h, std::string param_path, std::string model_path);
    ~Det();
    ncnn::Mat preprocess(cv::Mat img);
    void inference(std::string input_name, ncnn::Mat &input_, std::string output_name, int num_threads);
    void postprocess(int class_num, float confidence, float iou_thr);

    void generate_grids_and_stride();
    void yolox_decode(float prob_threshold);

    // static const char* class_name[] = { "person", "bicycle", "car", "motorcycle"};
    std::vector<ObjBox> out_boxes;
    std::vector<ObjBox> nms_boxes;
     

private:
    ncnn::Net net_;
    int input_w_;
    int input_h_;
    // ncnn::Mat input_;
    ncnn::Mat output_;

    const std::vector<int> strides{8, 16, 32};
    std::vector<GridAndStride> grid_strides;
    
};









}

