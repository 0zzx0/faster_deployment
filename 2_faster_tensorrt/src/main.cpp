#include "yolo.hpp"

using namespace std;


static const char* cocolabels[] = {
    "echinus", "starfish", "holothurian", "scallop"
};

static std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v){
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f*s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i) {
    case 0:r = v; g = t; b = p;break;
    case 1:r = q; g = v; b = p;break;
    case 2:r = p; g = v; b = t;break;
    case 3:r = p; g = q; b = v;break;
    case 4:r = t; g = p; b = v;break;
    case 5:r = v; g = p; b = q;break;
    default:r = 1; g = 1; b = 1;break;}
    return make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}

static std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id){
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}

static bool exists(const string& path){
    return access(path.c_str(), R_OK) == 0;
}

static string get_file_name(const string& path, bool include_suffix){

    if (path.empty()) return "";

    int p = path.rfind('/');
    int e = path.rfind('\\');
    p = std::max(p, e);
    p += 1;

    //include suffix
    if (include_suffix)
        return path.substr(p);

    int u = path.rfind('.');
    if (u == -1)
        return path.substr(p);

    if (u <= p) u = path.size();
    return path.substr(p, u - p);
}

static double timestamp_now_float() {
    return chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
}

static void inference_and_performance(int deviceid, const string& engine_file, YOLO::Mode mode, YOLO::YoloType type){

    int batch_size = 16;
    auto engine = YOLO::create_infer(engine_file, type, deviceid, batch_size, 0.4f, 0.5f);
    if(engine == nullptr){
        printf("Engine is nullptr\n");
        return;
    }

	vector<cv::String> files_;
	files_.reserve(10000);

    cv::glob("../inference/*.jpg", files_, true);
	vector<string> files(files_.begin(), files_.end());
    
    vector<cv::Mat> images;
    for(int i = 0; i < files.size(); ++i){
        auto image = cv::imread(files[i]);
        images.emplace_back(image);
    }

    // warmup
    vector<shared_future<YOLO::BoxArray>> boxes_array;
    for(int i = 0; i < 10; ++i)
        boxes_array = engine->commits(images);
    boxes_array.back().get();
    boxes_array.clear();
    
    /////////////////////////////////////////////////////////
    const int ntest = 100;
    auto begin_timer = timestamp_now_float();

    for(int i  = 0; i < ntest; ++i)
        boxes_array = engine->commits(images);

    // wait all result
    boxes_array.back().get();

    float inference_average_time = (timestamp_now_float() - begin_timer) / ntest / images.size();
    auto type_name = YOLO::type_name(type);
    auto mode_name = YOLO::mode_string(mode);
    printf("%s[%s] average: %.2f ms / image, FPS: %.2f\n", engine_file.c_str(), type_name, inference_average_time, 1000 / inference_average_time);

    string root = "../yolo_result";
    for(int i = 0; i < boxes_array.size(); ++i){

        auto& image = images[i];
        auto boxes  = boxes_array[i].get();
        
        for(auto& obj : boxes){
            uint8_t b, g, r;
            tie(b, g, r) = random_color(obj.class_label);
            cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

            auto name    = cocolabels[obj.class_label];
            auto caption = cv::format("%s %.2f", name, obj.confidence);
            int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(image, cv::Point(obj.left-3, obj.top-33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
            cv::putText(image, caption, cv::Point(obj.left, obj.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
        }

        string file_name = get_file_name(files[i], false);
        string save_path = cv::format("%s/%s.jpg", root.c_str(), file_name.c_str());
        printf("Save to %s, %d object, average time %.2f ms\n", save_path.c_str(), boxes.size(), inference_average_time);
        cv::imwrite(save_path, image);
    }
    engine.reset();
}

static void test(YOLO::YoloType type, YOLO::Mode mode, const string& model_name, bool is_onnxmodel){

    int deviceid = 0;
    auto mode_name = YOLO::mode_string(mode);
    YOLO::set_device(deviceid);

    const char* name = model_name.c_str();
    printf("===================== test %s %s %s ==================================\n", YOLO::type_name(type), mode_name, name);
    
    string model_file = cv::format("%s.%s.trtmodel", name, mode_name);
    if(is_onnxmodel){
        string onnx_file = cv::format("%s.onnx", name);
        int test_batch_size = 16;
        YOLO::compile(
            mode, type,                 // FP32、FP16、INT8
            test_batch_size,            // max batch size
            onnx_file,                  // source 
            model_file,                 // save to
            1 << 30,
            "inference"
        );
    }
    inference_and_performance(deviceid, model_file, mode, type);
}

void direct_test(){

    printf("TRTVersion: %s\n", YOLO::trt_version());
    
    int device_id = 0;
    string model = "yolox_s_dynamic";
    auto type = YOLO::YoloType::X;
    auto mode = YOLO::Mode::FP32;
    string onnx_file = cv::format("%s.onnx", model.c_str());
    string model_file = cv::format("%s.%s.trtmodel", model.c_str(), YOLO::mode_string(mode));
    YOLO::set_device(device_id);
    
    if(!exists(model_file) && !YOLO::compile(mode, type, 6, onnx_file, model_file, 1<<30, "inference")){
        printf("Compile failed\n");
        return;
    }

    float confidence_threshold = 0.4f;
    float nms_threshold = 0.5f;
    auto yolo = YOLO::create_infer(model_file, type, device_id, confidence_threshold, nms_threshold);
    if(yolo == nullptr){
        printf("Yolo is nullptr\n");
        return;
    }

    auto image = cv::imread("/home/zzx/Github/zzx_yolo/yolox_infer/imgs/000026.jpg");
    auto objs = yolo->commit(image).get();
    for(auto& obj : objs){
        uint8_t b, g, r;
        tie(b, g, r) = random_color(obj.class_label);
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

        auto name    = cocolabels[obj.class_label];
        auto caption = cv::format("%s %.2f", name, obj.confidence);
        int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(obj.left-3, obj.top-33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
        cv::putText(image, caption, cv::Point(obj.left, obj.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
    }

    printf("Save result to infer.jpg, %d objects\n", objs.size());
    cv::imwrite(cv::format("infer_%s.jpg", YOLO::trt_version()), image);
}

void only_trt_test(){

    printf("TRTVersion: %s\n", YOLO::trt_version());
    
    int device_id = 0;
    YOLO::set_device(device_id);
    auto type = YOLO::YoloType::X;
    const std::string model_file="/home/zzx/Github/zzx_yolo/yolox_infer/99_trt_new/yolox_b16.engine";
    // const std::string model_file="/home/zzx/Github/zzx_yolo/EXTRA_PKG/TensorRT-8.5.3.1/bin/yolox.engine";
    
    
    float confidence_threshold = 0.3f;
    float nms_threshold = 0.45f;
    auto yolo = YOLO::create_infer(model_file, type, device_id, 16, confidence_threshold, nms_threshold);
    if(yolo == nullptr){
        printf("Yolo is nullptr\n");
        return;
    }

    auto image = cv::imread("/home/zzx/Github/zzx_yolo/yolox_infer/imgs/000026.jpg");
    auto objs = yolo->commit(image).get();
    for(auto& obj : objs){
        // cout << obj.bottom <<" "<< obj.top <<" "<<obj.left<<" "<<obj.right << " " << obj.confidence <<endl;
        uint8_t b, g, r;
        tie(b, g, r) = random_color(obj.class_label);
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

        auto name    = cocolabels[obj.class_label];
        auto caption = cv::format("%s %.2f", name, obj.confidence);
        cout << caption << ": " << obj.bottom <<" "<< obj.top <<" "<<obj.left<<" "<<obj.right << endl;
        int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(obj.left-3, obj.top-33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
        cv::putText(image, caption, cv::Point(obj.left, obj.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
    }

    printf("Save result to infer.jpg, %d objects\n", objs.size());
    cv::imwrite(cv::format("infer_%s.jpg", YOLO::trt_version()), image);
}

static void zzx_test(YOLO::YoloType type, YOLO::Mode mode, const string& model_name){

    int deviceid = 0;
    auto mode_name = YOLO::mode_string(mode);
    YOLO::set_device(deviceid);

    printf("===================== test %s %s %s ==================================\n", YOLO::type_name(type), mode_name);
    
    
    inference_and_performance(deviceid, model_name, mode, type);
}

void zzx_test_video(){
    printf("TRTVersion: %s\n", YOLO::trt_version());
    
    int device_id = 0;
    YOLO::set_device(device_id);
    auto type = YOLO::YoloType::X;
    const std::string model_file="/home/zzx/Github/zzx_yolo/yolox_infer/99_trt_new/yolox_b16.engine";
    // const std::string model_file="/home/zzx/Github/zzx_yolo/EXTRA_PKG/TensorRT-8.5.3.1/bin/yolox.engine";
    
    float confidence_threshold = 0.3f;
    float nms_threshold = 0.45f;
    auto yolo = YOLO::create_infer(model_file, type, device_id, 16, confidence_threshold, nms_threshold);
    if(yolo == nullptr){
        printf("Yolo is nullptr\n");
        return;
    }

    const string video_file = "/home/zzx/Github/zzx_yolo/yolox_infer/99_trt_new/inference/2019-02-20_19-01-02to2019-02-20_19-01-13_1.avi";
    auto cap = cv::VideoCapture(video_file);
    cv::Mat frame;
    int frame_count = 0;
    
    vector<cv::Mat> images;

    auto begin_timer = timestamp_now_float();
    while (cap.isOpened()) {

        if(frame_count > 4000) break;
        frame_count++;

        // auto begin_timer1 = timestamp_now_float();
        // cap >> frame;
        // auto objs = yolo->commit(frame).get();
        if(images.size() != 2){
            cap >> frame;
            images.emplace_back(frame);
            continue;
        } 
        // auto begin_timer1 = timestamp_now_float();
        auto objs = yolo->commits(images);
        objs.back().get();
        images.clear();

        // float inference_average_time1 = (timestamp_now_float() - begin_timer1);
        // printf("time %.2f ms\n", inference_average_time1);
        // cout << objs.size() <<endl;
        // if(objs.size()){
        //     for(auto& obj : objs){
        //         cout << obj.bottom <<" "<< obj.top <<" "<<obj.left<<" "<<obj.right << " " << obj.confidence <<endl;
        //     }
        // }
    }
    float inference_average_time = (timestamp_now_float() - begin_timer) / frame_count;
    printf("average time %.2f ms\n", inference_average_time);

    cap.release();
}

int main(){
    zzx_test_video();
    // const std::string model_file="/home/zzx/Github/zzx_yolo/EXTRA_PKG/TensorRT-8.5.3.1/bin/yolox.engine";
    // const std::string model_file="/home/zzx/Github/zzx_yolo/yolox_infer/99_trt_new/yolox_b16.engine";
    // zzx_test(YOLO::YoloType::X, YOLO::Mode::FP16, model_file);

    // YOLO::compile(YOLO::Mode::FP16,YOLO::YoloType::X, 16, 
    //         "/home/zzx/Github/zzx_yolo/yolox_infer/utils/yolox.onnx", 
    //         "yolox_b16.engine", 
    //         1<<24);

    // test(YOLO::YoloType::X, YOLO::Mode::FP16, "yolox_s_dynamic", false);

    // direct_test();
    // only_trt_test();
    return 0;
}