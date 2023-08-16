#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "net.h"

#include "../base/tools.hpp"

namespace FasterNCNN {


// bboxes
struct ObjBox{

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


static float InterSectionArea(const ObjBox &a, const ObjBox &b);
static bool ScoreSort(ObjBox a, ObjBox b);
static void nms(std::vector<ObjBox> &src_boxes, std::vector<ObjBox> &dst_boxes, float threshold);


class postProcess {

public:
    enum class postProcessType : int{

            yolox   = 0,
            yolov8  = 1,
    };


public:
    postProcess(postProcessType type, float input_h, float input_w, float conf_thr, float nms_thr);
    ~postProcess() { };

    void forward(ncnn::Mat &output_);
    void yolox_generate_grids_and_stride();
    void yolox_decode(ncnn::Mat &output_);


protected:
    int input_h_;
    int input_w_;
    float conf_thr_;
    float nms_thr_;

    // std::vector<ObjBox> out_boxes;
    // std::vector<ObjBox> nms_boxes;

    const std::vector<int> strides{8, 16, 32};
    std::vector<GridAndStride> grid_strides;

public:
    std::vector<ObjBox> out_boxes;
    std::vector<ObjBox> nms_boxes;

};

}
