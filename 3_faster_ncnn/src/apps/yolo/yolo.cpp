#include "yolo.h"


namespace YoloNCNN{


bool InferImpl::startup(const std::string &param_path, const std::string &model_path, float confidence, float iou_thr){
    param_path_ = param_path;
    model_path_ = model_path;
    confidence_ = confidence;
    iou_thr_ = iou_thr;

    // 等待线程创建和里面的初始化完成
    return Det::startup();
}

void InferImpl::worker(std::promise<bool> &pro){

    input_w_ = 640;
    input_h_ = 640;
    input_name_ = "images";
    output_name_ = "output";

    net_.load_param(param_path_.c_str());
    net_.load_model(model_path_.c_str());
    postprocess_ = std::make_shared<postProcess>(postProcess::postProcessType::yolox, input_h_, input_w_, confidence_, iou_thr_);

    INFO("ncnn模型加载成功! ");

    pro.set_value(true); // satrtup 函数结束

    // std::vector<Job> fetch_jobs;
    Job fetch_job;
    while(get_job_and_wait(fetch_job)){
        
        input_ = fetch_job.input;
        forward();
        // output_
        fetch_job.pro->set_value(postprocess_->forward(output_ ));
    }

    INFO("推理结束！");
}


std::shared_future<std::vector<ObjBox>> InferImpl::commit(const cv::Mat &input){
    return Det::commit(input);
}


bool InferImpl::preprocess(Job &job, const cv::Mat &input) {
    int img_w = input.cols;
    int img_h = input.rows;

    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h){
        scale = (float)input_w_ / w;
        w = input_w_;
        h = h * scale;
    } else{
        scale = (float)input_h_ / h;
        h = input_h_;
        w = w * scale;
    }
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(input.data, ncnn::Mat::PIXEL_BGR, img_w, img_h, w, h);

    // pad to YOLOX_TARGET_SIZE rectangle
    int wpad = input_w_ - w;
    int hpad = input_h_ - h;

    ncnn::copy_make_border(in, job.input, 0, hpad, 0, wpad, ncnn::BORDER_CONSTANT, 114.f);
    // input_.substract_mean_normalize(mean_vals_, norm_vals_);
    return true;

}


std::shared_ptr<Infer> create_infer(const std::string &param_path, const std::string &model_path, float confidence, float iou_thr){
    std::shared_ptr<InferImpl> instance = std::make_shared<InferImpl>();
    if(!instance->startup(param_path, model_path, confidence, iou_thr)){
        instance.reset();
    }
    return instance;  // 创建子类对象 返回父类指针，这样实现封着。外部只能调用commit
}


} //end namespace