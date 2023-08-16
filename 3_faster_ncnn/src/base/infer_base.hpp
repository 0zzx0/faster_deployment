#pragma once

#include <queue>
#include <string>
#include <vector>
#include <thread>
#include <future>
#include <memory>
#include <condition_variable>

#include <opencv2/opencv.hpp>
#include "net.h"

#include "tools.hpp"

namespace FasterNCNN{


template<class Input, class Output>
class DetBase {

public:
    struct Job{
        ncnn::Mat input;
        Output output;
        std::shared_ptr<std::promise<Output>> pro;
    };


    virtual ~DetBase() {stop();};
    void stop() {
            run_ = false;
        cond_.notify_all();

        /// cleanup jobs
        {
            std::unique_lock<std::mutex> l(jobs_lock_);
            while(!jobs_.empty()){
                auto& item = jobs_.front();
                if(item.pro)
                    item.pro->set_value(Output());
                jobs_.pop();
            }
        };

        if(worker_){
            worker_->join();
            worker_.reset();
        }
    }

    // 启动 初始化线程 用一个promise等待worker中的初始化结束
    bool startup() {
        run_ = true;

        std::promise<bool> pro;
        worker_      = std::make_shared<std::thread>(&DetBase::worker, this, std::ref(pro));
        return pro.get_future().get();
    }

    // 工作线程(纯虚)
    virtual void worker(std::promise<bool>& result) = 0;
    // 预处理(纯虚)
    virtual bool preprocess(Job& job, const Input& input) = 0;



    virtual void forward() {
        auto ex = net_.create_extractor();
        // INFO("inputname: %s", input_name);
        // INFO("outputname: %s", output_name);
        // if(ncnn_use_vulkan_compute_) ex.set_vulkan_compute(true);
        ex.set_num_threads(ncnn_num_threads_);
        ex.input(input_name_.c_str(), input_);
        ex.extract(output_name_.c_str(), output_);
    }
    virtual std::shared_future<Output> commit(const Input& input) {
        Job job;
        job.pro = std::make_shared<std::promise<Output>>();
        if(!preprocess(job, input)){
            job.pro->set_value(Output());
            return job.pro->get_future();
        }
        
        ////////////////////上锁并且推进队列////////////////////////////
        {
            std::unique_lock<std::mutex> l(jobs_lock_);
            // jobs_.push(job);
            jobs_.emplace(job);
        };
        cond_.notify_one();
        return job.pro->get_future();
    }

    // 获取任务 等待之前的任务执行完毕
    virtual bool get_job_and_wait(Job& fetch_job) {
        std::unique_lock<std::mutex> l(jobs_lock_);
        cond_.wait(l, [&](){
            return !run_ || !jobs_.empty();
        });

        if(!run_) return false;
        
        fetch_job = std::move(jobs_.front());
        jobs_.pop();
        return true;
    }


protected:
    // ncnn
    ncnn::Net net_;
    int input_w_;
    int input_h_;
    std::string input_name_;
    std::string output_name_;
    ncnn::Mat input_;
    ncnn::Mat output_;
    int ncnn_num_threads_;
    bool ncnn_use_vulkan_compute_;

    // multi threads
    std::atomic<bool> run_;
    std::mutex jobs_lock_;
    std::queue<Job> jobs_;
    std::shared_ptr<std::thread> worker_;
    std::condition_variable cond_;

};

// 接口类 需要重写里面的纯虚函数
template<class Input, class Output>
class InferBase{
public:
    virtual std::shared_future<Output> commit(const Input &input) = 0;
};


}

