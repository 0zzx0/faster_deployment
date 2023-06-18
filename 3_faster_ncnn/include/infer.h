#pragma once

#include <future>
#include <memory>
#include <string>
#include <vector>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>


#include "opencv2/opencv.hpp"
#include "det.h"

namespace NCNN_DET{

struct Job
{
    std::shared_ptr<std::promise<std::vector<ObjBox>>> pro;
    ncnn::Mat input;
};


// 接口类 需要重写里面的纯虚函数
class Infer{
public:
    virtual std::shared_future<std::vector<ObjBox>> commit(cv::Mat &input) = 0;
};


// 推理
class InferImpl : public Infer{

public:
    virtual ~InferImpl();
    void stop();

    bool startup(const std::string &param_path,
                 const std::string &model_path, 
                 float confidence, float iou_thr);
    void worker(std::promise<bool> &pro);

    virtual std::shared_future<std::vector<ObjBox>> commit(cv::Mat &input) override;

private:
    std::atomic<bool> running_{false};
    std::string file_;
    std::thread worker_thread_;
    std::queue<Job> jobs_;
    std::mutex lock_;
    std::condition_variable cv_;

    Det *det_ = nullptr;

    std::string param_path_;
    std::string model_path_;
    int batch_size = 1;
    int input_w_ = 640;
    int input_h_ = 640;
    std::string input_name_ = "images";
    std::string output_name_ = "output";
    float confidence_;
    float iou_thr_;

    int infer_thread_ = 8;
    int class_num = 4;

};



std::shared_ptr<Infer> create_infer(const std::string &param_path, 
                                    const std::string &model_path, 
                                    float confidence,
                                    float iou_thr
                                    );

}
