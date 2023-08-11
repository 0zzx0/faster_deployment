#include "yolo.hpp"

using namespace std;

const int batch_size = 1;
const int deviceid = 0;
const float confidence_threshold = 0.5f;
const float nms_threshold = 0.65f;

const string model_file="../yolox_b16.engine";
auto type = YOLO::YoloType::X;

// 类别名 一般需要修改
static const char* classes[] = {
    "echinus", "starfish", "holothurian", "scallop"
};

// batch1 推理
void test_for_nsys() {

    printf("===================== test_for_nsys ==================================\n");
    printf("TRTVersion: %s\n", YOLO::trt_version());

    YOLO::set_device(deviceid);
    auto yolo = YOLO::create_infer(model_file, type, deviceid, batch_size, confidence_threshold, nms_threshold);
    if(yolo == nullptr) {
        printf("Yolo is nullptr\n");
        return;
    }

    auto image = cv::imread("../inference/1.jpg");
    
    // warmup
    for(int i = 0; i < 100; ++i){
        auto boxes_array = yolo->commit(image);
        boxes_array.get();
    }

    int ntest = 200;
    queue<shared_future<YOLO::BoxArray>> out_queue;
    auto start_time = std::chrono::system_clock::now();
    for(int i=0;i<ntest;i++){
        auto objs = yolo->commit(image);
        out_queue.emplace(objs);
        if(out_queue.size() <= 2){
            continue;
        }
        // 虽然看起来快了 但是相比于真实时间其实是需要分情况的，计算公式是下面
        // 可以看到如果模型比较大的时候这个方法才会受益

        auto res = out_queue.front().get();
        out_queue.pop();
    }
    while(!out_queue.empty()){
        auto res = out_queue.front().get();
        out_queue.pop();
    }

    auto end_time = std::chrono::system_clock::now();
    float inference_average_time = chrono::duration_cast<chrono::milliseconds>(end_time-start_time).count() / (ntest*1.0);
    printf("[zzx] average: %.2f ms / image, FPS: %.2f\n", inference_average_time, 1000 / inference_average_time);
}


// batch1 推理
void test_batch_1_img() {

    printf("===================== test_batch_1_imge ==================================\n");
    printf("TRTVersion: %s\n", YOLO::trt_version());

    YOLO::set_device(deviceid);
    auto yolo = YOLO::create_infer(model_file, type, deviceid, batch_size, confidence_threshold, nms_threshold);
    if(yolo == nullptr) {
        printf("Yolo is nullptr\n");
        return;
    }

    auto image = cv::imread("../inference/1.jpg");
    
    // warmup
    for(int i = 0; i < 500; ++i){
        auto boxes_array = yolo->commit(image);
        boxes_array.get();
    }

    int ntest = 1000;
    auto start_time = std::chrono::system_clock::now();
    for(int i=0;i<ntest;i++){
        auto objs = yolo->commit(image);
        auto res = objs.get();
    }

    auto end_time = std::chrono::system_clock::now();
    float inference_average_time = chrono::duration_cast<chrono::milliseconds>(end_time-start_time).count() / (ntest*1.0);
    printf("[zzx] average: %.2f ms / image, FPS: %.2f\n", inference_average_time, 1000 / inference_average_time);

    auto objs = yolo->commit(image);
    auto res = objs.get();
    for(auto& obj : res) {
        cout << obj.bottom <<" "<< obj.top <<" "<<obj.left<<" "<<obj.right << " " << obj.confidence <<endl;
    }
    /**
     * [zzx] average: 2.00 ms / image, FPS: 500.00
     */
}

// batch1 推理带queue
void test_batch_1_img_queue(int keep_queue_long){

    YOLO::set_device(deviceid);
    printf("===================== test_batch_1_img_queue ==================================\n");
    
    auto yolo = YOLO::create_infer(model_file, type, deviceid, batch_size, confidence_threshold, nms_threshold);
    if(yolo == nullptr){
        printf("Yolo is nullptr\n");
        return;
    }

    auto image = cv::imread("../inference/1.jpg");
    
    // warmup
    for(int i = 0; i < 500; ++i){
        auto boxes_array = yolo->commit(image);
        boxes_array.get();
    }

    queue<shared_future<YOLO::BoxArray>> out_queue;

    int ntest = 2000;
    auto start_time = std::chrono::system_clock::now();
    for(int i=0;i<ntest;i++){
        auto objs = yolo->commit(image);
        out_queue.emplace(objs);
        if(out_queue.size() <= keep_queue_long){
            continue;
        }
        // 虽然看起来快了 但是相比于真实时间其实是需要分情况的，计算公式是下面
        // 可以看到如果模型比较大的时候这个方法才会受益
        // ori: T + infer_time
        // zzx: T + (n-1)T
        auto res = out_queue.front().get();
        out_queue.pop();
    }
    while(!out_queue.empty()){
        auto res = out_queue.front().get();
        out_queue.pop();
    }

    auto end_time = std::chrono::system_clock::now();
    float inference_average_time = chrono::duration_cast<chrono::milliseconds>(end_time-start_time).count() / (ntest*1.0);
    printf("[zzx] average: %.2f ms / image, FPS: %.2f\n", inference_average_time, 1000 / inference_average_time);
    
    start_time = std::chrono::system_clock::now();
    for(int i=0;i<ntest;i++){
        auto objs = yolo->commit(image).get();
    }
    end_time = std::chrono::system_clock::now();
    inference_average_time = chrono::duration_cast<chrono::milliseconds>(end_time-start_time).count() / (ntest*1.0);
    printf("[ori] average: %.2f ms / image, FPS: %.2f\n", inference_average_time, 1000 / inference_average_time);

/*
    [zzx] average: 1.84 ms / image, FPS: 542.89
    [ori] average: 2.25 ms / image, FPS: 444.64
*/
}


static queue<shared_future<YOLO::BoxArray>> all_out_queue_;
static bool stop_=false;
static mutex all_mu_;
void test_batch_1_img_thread_work(){
    while(!stop_ || all_out_queue_.empty()){
        if(!all_out_queue_.empty()){
            std::shared_future<YOLO::BoxArray> res;
            {
                unique_lock<mutex> l(all_mu_);
                res = all_out_queue_.front();
                all_out_queue_.pop();
            }
            auto ans = res.get();
        }
    }
    return ;
}


void test_batch_1_img_thread(){
    
    YOLO::set_device(deviceid);
    printf("===================== zzx_test_batch_1_img ==================================\n");
    
    auto yolo = YOLO::create_infer(model_file, type, deviceid, batch_size, confidence_threshold, nms_threshold);
    if(yolo == nullptr){
        printf("Yolo is nullptr\n");
        return;
    }

    auto image = cv::imread("../inference/1.jpg");
    
    // warmup
    shared_future<YOLO::BoxArray> boxes_array;
    for(int i = 0; i < 500; ++i){
        boxes_array = yolo->commit(image);
        boxes_array.get();
    }

    int ntest = 1000;
    auto image_now = cv::imread("../inference/1.jpg");
    
    auto start_time = std::chrono::system_clock::now();
    thread mywork(test_batch_1_img_thread_work);
    for(int i=0;i<ntest;i++){
        auto objs = yolo->commit(image_now);
        {
            unique_lock<mutex> l(all_mu_);
            all_out_queue_.emplace(objs);
        }
    }
    {
        unique_lock<mutex> l(all_mu_);
        stop_ = true;
    }
    mywork.join();

    auto end_time = std::chrono::system_clock::now();
    float inference_average_time = chrono::duration_cast<chrono::milliseconds>(end_time-start_time).count() / (ntest*1.0);
    printf("[zzx] average: %.2f ms / image, FPS: %.2f\n", inference_average_time, 1000 / inference_average_time);
    
}


void test_batch_1_video(){
   
    YOLO::set_device(deviceid);
    printf("===================== zzx_v1_test_batch_1_video ==================================\n");
    
    auto yolo = YOLO::create_infer(model_file, type, deviceid, batch_size, confidence_threshold, nms_threshold);
    if(yolo == nullptr){
        printf("Yolo is nullptr\n");
        return;
    }

    auto image = cv::imread("../inference/000026.jpg");
    
    // warmup
    shared_future<YOLO::BoxArray> boxes_array;
    for(int i = 0; i < 100; ++i){
        boxes_array = yolo->commit(image);
        boxes_array.get();
    }

    const string video_file = "../inference/2019-02-20_19-01-02to2019-02-20_19-01-13_1.avi";
    auto cap = cv::VideoCapture(video_file);
    cv::Mat frame;
    int frame_count = 0;
    int ntest = 1000;

    queue<shared_future<YOLO::BoxArray>> out_queue;

    auto start_time = std::chrono::system_clock::now();
    while (cap.isOpened()) {
        if(frame_count > ntest) break;
        frame_count++;
        cap >> frame;

        auto objs = yolo->commit(frame);
        out_queue.emplace(objs);

        if(out_queue.size() < 2){
            continue;
        }
        auto res = out_queue.front().get();
        out_queue.pop();
    }
    while(!out_queue.empty()){
        auto res = out_queue.front().get();
        out_queue.pop();
    }

    auto end_time = std::chrono::system_clock::now();
    float inference_average_time = chrono::duration_cast<chrono::milliseconds>(end_time-start_time).count() / ntest;
    printf("[zzx] average: %.2f ms / image, FPS: %.2f\n", inference_average_time, 1000 / inference_average_time);
    cap.release();

}


int main(){
    test_for_nsys();
    // test_batch_1_img();
    // test_batch_1_img_queue(2);
    // test_batch_1_img_thread();
    return 0;
}