#ifndef INFER_BASE_HPP
#define INFER_BASE_HPP

#include "memory_tensor.hpp"
#include "monopoly_accocator.hpp"
#include "cuda_kernel.cuh"

#include <string>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>

#include <NvInfer.h>


namespace YOLO{

using namespace nvinfer1;
using namespace cv;

/////////////////////////////////TRT logger部分/////////////////////////////////
class Logger : public ILogger {
public:
    virtual void log(Severity severity, const char* msg) noexcept override {

        if (severity == Severity::kINTERNAL_ERROR) {
            INFOE("NVInfer INTERNAL_ERROR: %s", msg);
            abort();
        }else if (severity == Severity::kERROR) {
            INFOE("NVInfer: %s", msg);
        }
        else  if (severity == Severity::kWARNING) {
            INFOW("NVInfer: %s", msg);
        }
        else  if (severity == Severity::kINFO) {
            INFOD("NVInfer: %s", msg);
        }
        else {
            INFOD("%s", msg);
        }
    }
};
static Logger gLogger;

// 销毁tensorrt中间指针对象的函数模板
template<typename _T>
static void destroy_nvidia_pointer(_T* ptr) {
    if (ptr) ptr->destroy();
}


/*
    trt运行过程中的几个指针对象打包了
    方便一起创建和销毁
*/
class EngineContext {
public:
    virtual ~EngineContext() { destroy(); }

    // 设置stream 如果已经存在销毁旧的，添加新的
    void set_stream(cudaStream_t stream){
        if(owner_stream_){
            if (stream_) {cudaStreamDestroy(stream_);}
            owner_stream_ = false;
        }
        stream_ = stream;
    }

    // 使用智能指针创建runtime engine context和初始化stream
    bool build_model(const void* pdata, size_t size) {
        destroy();

        if(pdata == nullptr || size == 0)
            return false;

        owner_stream_ = true;
        checkCudaRuntime(cudaStreamCreate(&stream_));
        if(stream_ == nullptr)
            return false;

        runtime_ = shared_ptr<IRuntime>(createInferRuntime(gLogger), destroy_nvidia_pointer<IRuntime>);
        if (runtime_ == nullptr)
            return false;

        engine_ = shared_ptr<ICudaEngine>(runtime_->deserializeCudaEngine(pdata, size, nullptr), destroy_nvidia_pointer<ICudaEngine>);
        if (engine_ == nullptr)
            return false;

        //runtime_->setDLACore(0);
        context_ = shared_ptr<IExecutionContext>(engine_->createExecutionContext(), destroy_nvidia_pointer<IExecutionContext>);
        return context_ != nullptr;
    }

private:
    // 销毁这些指针 通过让智能指针引用计数减一
    void destroy() {
        context_.reset();
        engine_.reset();
        runtime_.reset();

        if(owner_stream_){
            if (stream_) {cudaStreamDestroy(stream_);}
        }
        stream_ = nullptr;
    }

public:
    cudaStream_t stream_ = nullptr;
    bool owner_stream_ = false;
    shared_ptr<IExecutionContext> context_;
    shared_ptr<ICudaEngine> engine_;
    shared_ptr<IRuntime> runtime_ = nullptr;
};


/*
    推理引擎的创建和推理
    可以获取推理模型的各类输入输出信息
*/
class TRTInferImpl{
public:
    virtual ~TRTInferImpl();
    bool load(const std::string& file, int batch_size);     
    bool load_from_memory(const void* pdata, size_t size);
    void destroy();

    void forward(bool sync);

    int get_max_batch_size();
    cudaStream_t get_stream();
    void set_stream(cudaStream_t stream);
    void synchronize();
    size_t get_device_memory_size();
    std::shared_ptr<MixMemory> get_workspace();
    std::shared_ptr<Tensor> input(int index = 0);
    std::string get_input_name(int index = 0);
    std::shared_ptr<Tensor> output(int index = 0);
    std::string get_output_name(int index = 0);
    std::shared_ptr<Tensor> tensor(const std::string& name);
    bool is_output_name(const std::string& name);
    bool is_input_name(const std::string& name);
    void set_input (int index, std::shared_ptr<Tensor> tensor);
    void set_output(int index, std::shared_ptr<Tensor> tensor);
    std::shared_ptr<std::vector<uint8_t>> serial_engine();

    void print();

    int num_output();
    int num_input();
    int device();

private:
    void build_engine_input_and_outputs_mapper();

private:
    std::vector<std::shared_ptr<Tensor>> inputs_;  
    std::vector<std::shared_ptr<Tensor>> outputs_;
    std::vector<int> inputs_map_to_ordered_index_;
    std::vector<int> outputs_map_to_ordered_index_;
    std::vector<std::string> inputs_name_;
    std::vector<std::string> outputs_name_;
    std::vector<std::shared_ptr<Tensor>> orderdBlobs_;
    std::map<std::string, int> blobsNameMapper_;
    std::shared_ptr<EngineContext> context_;
    std::vector<void*> bindingsPtr_;
    std::shared_ptr<MixMemory> workspace_;
    int device_ = 0;
    int batch_max_size_ = 1;
};



/* 
    异步线程安全的推理器(这是一个虚基类 子类至少重写preprocess work)
    通过异步线程启动，使得调用方允许任意线程调用把图像做输入，并通过future来获取异步结果
    因为是一个模板类，所以懒得在类外实现了，在infer_base同时定义和实现。
*/
template<class Input, class Output, class StartParam=std::tuple<std::string, int>, class JobAdditional=int>
class ThreadSafedAsyncInfer{
public:
    // Job数据类型。
    struct Job{
        Input input;
        Output output;
        JobAdditional additional;
        MonopolyAllocator<Tensor>::MonopolyDataPointer mono_tensor;
        std::shared_ptr<std::promise<Output>> pro;
    };

    virtual ~ThreadSafedAsyncInfer(){
        stop();
    }

    // 停止 由析构函数调用
    void stop(){
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
    bool startup(const StartParam& param){
        run_ = true;

        std::promise<bool> pro;
        start_param_ = param;
        worker_      = std::make_shared<std::thread>(&ThreadSafedAsyncInfer::worker, this, std::ref(pro));
        return pro.get_future().get();
    }

    // 单输入commit 先预处理input 然后上锁推进工作队列 cond_ 提醒 然后开始等待output
    virtual std::shared_future<Output> commit(const Input& input){

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
    
    // vector 输入commit
    virtual std::vector<std::shared_future<Output>> commits(const std::vector<Input>& inputs){

        int batch_size = std::min((int)inputs.size(), this->tensor_allocator_->capacity());
        std::vector<Job> jobs(inputs.size());
        std::vector<std::shared_future<Output>> results(inputs.size());

        int nepoch = (inputs.size() + batch_size - 1) / batch_size;
        for(int epoch = 0; epoch < nepoch; ++epoch){
            int begin = epoch * batch_size;
            int end   = std::min((int)inputs.size(), begin + batch_size);

            for(int i = begin; i < end; ++i){
                Job& job = jobs[i];
                job.pro = std::make_shared<std::promise<Output>>();
                if(!preprocess(job, inputs[i])){
                    job.pro->set_value(Output());
                }
                results[i] = job.pro->get_future();
            }

            ///////////////////////////////////////////////////////////
            {
                std::unique_lock<std::mutex> l(jobs_lock_);
                for(int i = begin; i < end; ++i){
                    jobs_.emplace(std::move(jobs[i]));
                };
            }
            cond_.notify_one(); 
        }
        return results;
    }

protected:
    // 工作线程(纯虚)
    virtual void worker(std::promise<bool>& result) = 0;
    // 预处理(纯虚)
    virtual bool preprocess(Job& job, const Input& input) = 0;
    
    // 获取任务组 等待之前的任务执行完毕
    virtual bool get_jobs_and_wait(std::vector<Job>& fetch_jobs, int max_size){

        std::unique_lock<std::mutex> l(jobs_lock_);
        cond_.wait(l, [&](){
            return !run_ || !jobs_.empty();
        }); // 当前run=true 且 job为empty(队列中的任务做完)的时候才会等待

        if(!run_) return false;
        
        fetch_jobs.clear();
        for(int i = 0; i < max_size && !jobs_.empty(); ++i){
            fetch_jobs.emplace_back(std::move(jobs_.front()));
            jobs_.pop();
        }
        return true;
    }

    // 获取任务 等待之前的任务执行完毕
    virtual bool get_job_and_wait(Job& fetch_job){

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
    StartParam start_param_;
    std::atomic<bool> run_;
    std::mutex jobs_lock_;
    std::queue<Job> jobs_;
    std::shared_ptr<std::thread> worker_;
    std::condition_variable cond_;
    std::shared_ptr<MonopolyAllocator<Tensor>> tensor_allocator_;
};

// 产生一个trt推理的智能指针 参数是序列化文件路径
std::shared_ptr<TRTInferImpl> load_infer(const string& file, int batch_size);

};

#endif