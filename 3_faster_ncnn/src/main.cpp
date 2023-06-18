#include <iostream>
#include <opencv2/opencv.hpp>
#include<chrono>

#include "infer.h"

using namespace std;

void t2(){
    string param_path = "/home/zzx/Github/zzx_yolo/yolox_infer/4_ncnn/model.param";
    string model_path = "/home/zzx/Github/zzx_yolo/yolox_infer/4_ncnn/model.bin";

    auto infer = NCNN_DET::create_infer(param_path, model_path, 0.5, 0.65); 

    if (infer == nullptr){
        printf("Infer is nullptr.\n");
        return ;
    }
    
    string img_path = "/home/zzx/Github/zzx_yolo/yolox_infer/imgs/000026.jpg";
    cv::Mat img = cv::imread(img_path);

    queue<std::shared_future<std::vector<NCNN_DET::ObjBox>>> out_queue;

    auto start = std::chrono::system_clock::now();
    for(int i=0;i<100;i++){
        // img_queue.push(img);
        // auto start = std::chrono::system_clock::now();
        auto fut = infer->commit(img);     // 将任务提交给推理器（推理器执行commit)
        // auto end = std::chrono::system_clock::now();
        // cout << chrono::duration_cast<chrono::milliseconds>(end - start).count()<< "ms" << endl;

        out_queue.push(fut);

        if(out_queue.size() <= 30){
            continue;
        }
        auto res = out_queue.front().get();
        out_queue.pop();
    }
    while(!out_queue.empty()){
        auto res = out_queue.front().get();
        out_queue.pop();
    }
    
    auto end = std::chrono::system_clock::now();
    cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() / 100.0f<< "ms" << endl;

    // cout << res.size() << endl;
    // for (size_t i = 0; i < res.size(); i++){
    //     NCNN_DET::ObjBox box = res[i];
    //     cout<<box.score<<endl;
    //     cv::rectangle(img, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2),
    //               cv::Scalar(0, 0, 255), 2);
    //     // cv::putText(img, pred->class_names[box.category], cv::Point(box.x1,
    //     // box.y1),
    //     //             cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 2);
    // }
    // cv::imwrite("result_test.jpg", img);

    return ;
}

void t1(){
    string param_path = "/home/zzx/Github/zzx_yolo/yolox_infer/4_ncnn/model.param";
    string model_path = "/home/zzx/Github/zzx_yolo/yolox_infer/4_ncnn/model.bin";

    auto infer = NCNN_DET::create_infer(param_path, model_path, 0.5, 0.65); 

    if (infer == nullptr){
        printf("Infer is nullptr.\n");
        return ;
    }

    string img_path = "/home/zzx/Github/zzx_yolo/yolox_infer/imgs/000026.jpg";
    cv::Mat img = cv::imread(img_path);

    auto start = std::chrono::system_clock::now();
    for(int i=0;i<100;i++){
        auto fut = infer->commit(img);     // 将任务提交给推理器（推理器执行commit)
        vector<NCNN_DET::ObjBox> res = fut.get(); // 等待结果
    }
    
    auto end = std::chrono::system_clock::now();
    cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() / 100.0f<< "ms" << endl;

    // cout << res.size() << endl;
    // for (size_t i = 0; i < res.size(); i++){
    //     NCNN_DET::ObjBox box = res[i];
    //     cout<<box.score<<endl;
    //     cv::rectangle(img, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2),
    //               cv::Scalar(0, 0, 255), 2);
    //     // cv::putText(img, pred->class_names[box.category], cv::Point(box.x1,
    //     // box.y1),
    //     //             cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 2);
    // }
    // cv::imwrite("result_test.jpg", img);
    return ;

}

int main(){
    t1();
    t2();

}

