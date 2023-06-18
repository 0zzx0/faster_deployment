#include "infer.h"
#include "mylogger.hpp"

namespace NCNN_DET{

InferImpl::~InferImpl(){
    stop();
}

void InferImpl::stop(){
    if(running_){
        running_ = false;
        cv_.notify_one();
    }
    if(worker_thread_.joinable()){
        worker_thread_.join();
    }
    if(det_!=nullptr){
        INFOE("det_ 没有被正确销毁！");
    }
}

bool InferImpl::startup(const std::string &param_path, const std::string &model_path, float confidence, float iou_thr){
    param_path_ = param_path;
    model_path_ = model_path;
    confidence_ = confidence;
    iou_thr_ = iou_thr;
    running_ = true;

    // 使用promise获取是否初始化成功
    // 在线程内初始化 内存申请和释放都在线程里面
    std::promise<bool> pro;
    worker_thread_ = std::thread(&InferImpl::worker, this, std::ref(pro));

    // 等待线程创建和里面的初始化完成
    return pro.get_future().get();
}

void InferImpl::worker(std::promise<bool> &pro){
    det_ = new Det(input_w_, input_h_, param_path_, model_path_);

    if(det_ == nullptr){
        pro.set_value(false);
        INFOE("load ncnn model failed.");
        return ;
    }
    pro.set_value(true); // satrtup 函数结束

    std::vector<Job> fetch_jobs;
    while(running_){
        {
            std::unique_lock<std::mutex> l(lock_);
            cv_.wait(l, [&](){return !running_ || !jobs_.empty();});

            if(!running_) break;

            // 把队列中的图片搬到fetch_jobs
            for(int i=0; i<batch_size; ++i){
                fetch_jobs.emplace_back(std::move(jobs_.front()));
                jobs_.pop();

            }
        }
        for(auto &job : fetch_jobs){
            det_->inference(input_name_, job.input, output_name_, infer_thread_);
            det_->postprocess(class_num, confidence_, iou_thr_);
            job.pro->set_value(det_->nms_boxes);
        }
        fetch_jobs.clear();
    }

    delete det_;
    det_ = nullptr;
    INFO("推理结束！");
}


std::shared_future<std::vector<ObjBox>> InferImpl::commit(cv::Mat &input){
    Job job;
    job.input = det_->preprocess(input);
    job.pro.reset(new std::promise<std::vector<ObjBox>>);

    std::shared_future<std::vector<ObjBox>> fu = job.pro->get_future();
    {
        std::unique_lock<std::mutex> l(lock_);
        jobs_.emplace(std::move(job));
    }
    cv_.notify_one();
    return fu;
}


std::shared_ptr<Infer> create_infer(const std::string &param_path, const std::string &model_path, float confidence, float iou_thr){
    std::shared_ptr<InferImpl> instance(new InferImpl());
    if(!instance->startup(param_path, model_path, confidence, iou_thr)){
        instance.reset();
    }
    return instance;  // 穿件子类对象 返回父类指针，这样实现封着。外部只能调用commit
}


} //end namespace