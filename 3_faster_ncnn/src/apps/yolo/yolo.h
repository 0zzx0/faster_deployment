#pragma once

#include "opencv2/opencv.hpp"

#include "../common.h"
#include "../../base/infer_base.hpp"
#include "../../base/tools.hpp"


namespace YoloNCNN{

using namespace FasterNCNN;


using Infer = InferBase<cv::Mat, std::vector<ObjBox>>;
using Det = DetBase<cv::Mat, std::vector<ObjBox>>;
// 推理
class InferImpl : public Infer, Det{

public:


    bool startup(const std::string &param_path,
                 const std::string &model_path, 
                 float confidence, float iou_thr);
    virtual void worker(std::promise<bool> &pro) override;
    virtual bool preprocess(Job &job, const cv::Mat &input) override;

    virtual std::shared_future<std::vector<ObjBox>> commit(const cv::Mat &input) override;

private:

    std::string param_path_;
    std::string model_path_;
    float confidence_;
    float iou_thr_;

    std::shared_ptr<postProcess> postprocess_;
    std::vector<ObjBox> results_;

    int infer_thread_ = 8;
    int class_num = 4;

};



std::shared_ptr<Infer> create_infer(const std::string &param_path, 
                                    const std::string &model_path, 
                                    float confidence,
                                    float iou_thr
                                    );

}
