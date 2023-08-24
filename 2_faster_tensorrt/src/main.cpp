#include "./apps/yolo/yolo.h"
#include "./apps/rtdetr/rtdetr.h"

using namespace std;

const int batch_size = 1;
const int deviceid = 0;
const float confidence_threshold = 0.5f;
const float nms_threshold = 0.65f;

const string model_file = "../yolox_b16.engine";
auto type = YOLO::YoloType::X;

// 类别名 一般需要修改
static const char* classes[] = {"echinus", "starfish", "holothurian", "scallop"};
static void printBox(vector<YOLO::Box>& boxs) {
    printf("obj nums: %d \n", boxs.size());
    for(int i = 0; i < boxs.size(); ++i) {
        printf("obj[%d], socre: %.3f, id: %d, location: [%f, %f, %f, %f]\n", i, boxs[i].confidence,
               boxs[i].class_label, boxs[i].left, boxs[i].top, boxs[i].right, boxs[i].bottom);
    }
}

// batch1 推理
void test_for_nsys() {
    printf("===================== test_for_nsys ==================================\n");
    printf("TRTVersion: %s\n", YOLO::trt_version());

    YOLO::set_device(deviceid);
    auto yolo = YOLO::create_infer(model_file, type, deviceid, batch_size, confidence_threshold,
                                   nms_threshold);
    if(yolo == nullptr) {
        printf("Yolo is nullptr\n");
        return;
    }

    auto image = cv::imread("../inference/1.jpg");

    // warmup
    for(int i = 0; i < 500; ++i) {
        auto boxes_array = yolo->commit(image);
        boxes_array.get();
    }
    int ntest = 1000;
    queue<shared_future<YOLO::BoxArray>> out_queue;
    auto start_time = std::chrono::system_clock::now();
    for(int i = 0; i < ntest; i++) {
        auto objs = yolo->commit(image);
        out_queue.emplace(objs);
        if(out_queue.size() <= 1) {
            continue;
        }
        auto res = out_queue.front().get();
        out_queue.pop();
        // 1.42ms   704.47fps
    }
    while(!out_queue.empty()) {
        auto res = out_queue.front().get();
        out_queue.pop();
    }

    // int ntest = 1000;
    // auto start_time = std::chrono::system_clock::now();
    for(int i = 0; i < ntest; i++) {
        auto objs = yolo->commit(image).get();
        // 1.86ms 538.79fps
    }
    auto end_time = std::chrono::system_clock::now();
    float inference_average_time =
        chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() / (ntest * 1.0);
    printf("[zzx] average: %.2f ms / image, FPS: %.2f\n", inference_average_time,
           1000 / inference_average_time);
}

// batch1 推理
void test_batch_1_img() {
    printf("===================== test_batch_1_imge ==================================\n");
    printf("TRTVersion: %s\n", YOLO::trt_version());

    YOLO::set_device(deviceid);
    auto yolo = YOLO::create_infer(model_file, type, deviceid, batch_size, confidence_threshold,
                                   nms_threshold);
    if(yolo == nullptr) {
        printf("Yolo is nullptr\n");
        return;
    }

    auto image = cv::imread("../inference/1.jpg");

    // warmup
    for(int i = 0; i < 500; ++i) {
        auto boxes_array = yolo->commit(image);
        boxes_array.get();
    }

    int ntest = 200;
    auto start_time = std::chrono::system_clock::now();
    for(int i = 0; i < ntest; i++) {
        auto objs = yolo->commit(image);
        auto res = objs.get();
    }

    auto end_time = std::chrono::system_clock::now();
    float inference_average_time =
        chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() / (ntest * 1.0);
    printf("[zzx] average: %.2f ms / image, FPS: %.2f\n", inference_average_time,
           1000 / inference_average_time);

    auto objs = yolo->commit(image);
    auto res = objs.get();
    for(auto& obj : res) {
        cout << obj.bottom << " " << obj.top << " " << obj.left << " " << obj.right << " "
             << obj.confidence << endl;
    }
    /**
     * [zzx] average: 2.28 ms / image, FPS: 438.60
     */
}

// batch1 推理带queue
void test_batch_1_img_queue(int keep_queue_long) {
    YOLO::set_device(deviceid);
    printf("===================== test_batch_1_img_queue ==================================\n");

    auto yolo = YOLO::create_infer(model_file, type, deviceid, batch_size, confidence_threshold,
                                   nms_threshold);
    if(yolo == nullptr) {
        printf("Yolo is nullptr\n");
        return;
    }

    auto image = cv::imread("../inference/1.jpg");

    // warmup
    for(int i = 0; i < 500; ++i) {
        auto boxes_array = yolo->commit(image);
        boxes_array.get();
    }

    queue<shared_future<YOLO::BoxArray>> out_queue;

    int ntest = 2000;
    auto start_time = std::chrono::system_clock::now();
    for(int i = 0; i < ntest; i++) {
        auto objs = yolo->commit(image);
        out_queue.emplace(objs);
        if(out_queue.size() < keep_queue_long) {
            continue;
        }
        // 虽然看起来快了 但是相比于真实时间其实是需要分情况的，计算公式是下面
        // 可以看到如果模型比较大的时候这个方法才会受益
        // ori: T + infer_time
        // zzx: T + (n-1)T
        auto res = out_queue.front().get();
        out_queue.pop();
    }
    while(!out_queue.empty()) {
        auto res = out_queue.front().get();
        out_queue.pop();
    }

    auto end_time = std::chrono::system_clock::now();
    float inference_average_time =
        chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() / (ntest * 1.0);
    printf("[zzx] average: %.2f ms / image, FPS: %.2f\n", inference_average_time,
           1000 / inference_average_time);

    start_time = std::chrono::system_clock::now();
    for(int i = 0; i < ntest; i++) {
        auto objs = yolo->commit(image).get();
    }
    end_time = std::chrono::system_clock::now();
    inference_average_time =
        chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() / (ntest * 1.0);
    printf("[ori] average: %.2f ms / image, FPS: %.2f\n", inference_average_time,
           1000 / inference_average_time);

    /*
        [zzx] average: 1.84 ms / image, FPS: 542.89
        [zzx_multi_stream] average: 1.41 ms / image, FPS: 709.98
        [ori] average: 2.25 ms / image, FPS: 444.64
    */
}

static queue<shared_future<YOLO::BoxArray>> all_out_queue_;
static bool stop_ = false;
static mutex all_mu_;
void test_batch_1_img_thread_work() {
    while(!stop_ || all_out_queue_.empty()) {
        if(!all_out_queue_.empty()) {
            std::shared_future<YOLO::BoxArray> res;
            {
                unique_lock<mutex> l(all_mu_);
                res = all_out_queue_.front();
                all_out_queue_.pop();
            }
            auto ans = res.get();
        }
    }
    return;
}

void test_batch_1_img_thread() {
    YOLO::set_device(deviceid);
    printf("===================== zzx_test_batch_1_img ==================================\n");

    auto yolo = YOLO::create_infer(model_file, type, deviceid, batch_size, confidence_threshold,
                                   nms_threshold);
    if(yolo == nullptr) {
        printf("Yolo is nullptr\n");
        return;
    }

    auto image = cv::imread("../inference/1.jpg");

    // warmup
    shared_future<YOLO::BoxArray> boxes_array;
    for(int i = 0; i < 500; ++i) {
        boxes_array = yolo->commit(image);
        boxes_array.get();
    }

    int ntest = 1000;
    auto image_now = cv::imread("../inference/1.jpg");

    auto start_time = std::chrono::system_clock::now();
    thread mywork(test_batch_1_img_thread_work);
    for(int i = 0; i < ntest; i++) {
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
    float inference_average_time =
        chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() / (ntest * 1.0);
    printf("[zzx] average: %.2f ms / image, FPS: %.2f\n", inference_average_time,
           1000 / inference_average_time);
}

void test_batch_1_video() {
    YOLO::set_device(deviceid);
    printf("===================== zzx_v1_test_batch_1_video ==================================\n");

    auto yolo = YOLO::create_infer(model_file, type, deviceid, batch_size, confidence_threshold,
                                   nms_threshold);
    if(yolo == nullptr) {
        printf("Yolo is nullptr\n");
        return;
    }

    auto image = cv::imread("../inference/000026.jpg");

    // warmup
    shared_future<YOLO::BoxArray> boxes_array;
    for(int i = 0; i < 100; ++i) {
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
    while(cap.isOpened()) {
        if(frame_count > ntest) break;
        frame_count++;
        cap >> frame;

        auto objs = yolo->commit(frame);
        out_queue.emplace(objs);

        if(out_queue.size() < 2) {
            continue;
        }
        auto res = out_queue.front().get();
        out_queue.pop();
    }
    while(!out_queue.empty()) {
        auto res = out_queue.front().get();
        out_queue.pop();
    }

    auto end_time = std::chrono::system_clock::now();
    float inference_average_time =
        chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() / ntest;
    printf("[zzx] average: %.2f ms / image, FPS: %.2f\n", inference_average_time,
           1000 / inference_average_time);
    cap.release();
}

// batch1 推理
void test_for_v8() {
    printf("===================== test_for_nsys ==================================\n");
    printf("TRTVersion: %s\n", YOLO::trt_version());

    YOLO::set_device(deviceid);
    // const string model_file1="../../5_yolov8/trt_weights/v8n.trt";
    const string model_file1 = "../../5_yolov8/v8_transpose.trt";
    auto type1 = YOLO::YoloType::V8;
    auto yolo = YOLO::create_infer(model_file1, type1, deviceid, batch_size, confidence_threshold,
                                   nms_threshold);
    if(yolo == nullptr) {
        printf("Yolo is nullptr\n");
        return;
    }

    auto image = cv::imread("../inference/1.jpg");
    auto boxes_array = yolo->commit(image).get();
    printBox(boxes_array);
    // cout << boxes_array.size() << endl;

    YOLO::set_device(deviceid);
    auto yolo1 = YOLO::create_infer(model_file, type, deviceid, batch_size, confidence_threshold,
                                    nms_threshold);
    boxes_array = yolo1->commit(image).get();
    printBox(boxes_array);
}

// batch1 推理
void test_for_rtdetr() {
    printf("========================= test_for_rtdetr ===============================\n");
    printf("TRTVersion: %s\n", RTDETR::trt_version());

    RTDETR::set_device(deviceid);
    const string model_file1 =
        "/home/zzx/Experiment/PaddleDetection/0zzx/new/rtdetr_r18vd_6x_coco.trt";
    auto rtdetr = RTDETR::create_infer(model_file1, deviceid, batch_size, 0.5f);
    if(rtdetr == nullptr) {
        printf("rtdetr is nullptr\n");
        return;
    }

    auto image = cv::imread("/home/zzx/Experiment/PaddleDetection/demo/000000014439.jpg");
    auto boxes_array = rtdetr->commit(image).get();
    printBox(boxes_array);
    // cout << boxes_array.size() << endl;

    // warmup
    for(int i = 0; i < 500; ++i) {
        auto boxes_array = rtdetr->commit(image);
        boxes_array.get();
    }
    int ntest = 2000;
    queue<shared_future<RTDETR::BoxArray>> out_queue;
    auto start_time = std::chrono::system_clock::now();
    for(int i = 0; i < ntest; i++) {
        auto objs = rtdetr->commit(image);
        out_queue.emplace(objs);
        if(out_queue.size() <= 1) {
            continue;
        }
        auto res = out_queue.front().get();
        out_queue.pop();
    }
    while(!out_queue.empty()) {
        auto res = out_queue.front().get();
        out_queue.pop();
    }

    // auto start_time = std::chrono::system_clock::now();
    // for(int i = 0; i < ntest; i++) {
    //     auto objs = rtdetr->commit(image).get();
    // }
    auto end_time = std::chrono::system_clock::now();
    float inference_average_time =
        chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() / (ntest * 1.0);
    printf("[zzx] average: %.2f ms / image, FPS: %.2f\n", inference_average_time,
           1000 / inference_average_time);
}

int main() {
    // test_for_nsys();
    // test_batch_1_img();
    // test_batch_1_img_queue(2);
    // test_batch_1_img_thread();
    // test_for_v8();
    test_for_rtdetr();
    return 0;
}