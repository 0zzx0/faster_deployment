#include <chrono>

#include "apps/yolo/yolo.h"

using namespace std;

// 打印结果因袭
static void printBox(vector<YoloNCNN::ObjBox> &boxs) {
    printf("obj nums: %d \n", boxs.size());
    for(int i=0;i<boxs.size();++i) {
        printf("obj[%d], socre: %.3f, id: %d, location: [%d, %d, %d, %d]\n", 
                i, boxs[i].score, boxs[i].category, boxs[i].x1, boxs[i].x2, boxs[i].y1, boxs[i].y2);
    }
}


void t2(){
    string param_path = "../model.param";
    string model_path = "../model.bin";

    auto infer = YoloNCNN::create_infer(param_path, model_path, 0.5, 0.65); 

    if (infer == nullptr){
        printf("Infer is nullptr.\n");
        return ;
    }
    
    string img_path = "../img/000026.jpg";
    cv::Mat img = cv::imread(img_path);

    queue<std::shared_future<std::vector<YoloNCNN::ObjBox>>> out_queue;

    auto start = std::chrono::system_clock::now();
    for(int i=0;i<100;i++){

        // auto start = std::chrono::system_clock::now();
        auto fut = infer->commit(img);     // 任务提交
        // auto end = std::chrono::system_clock::now();
        // cout << chrono::duration_cast<chrono::milliseconds>(end - start).count()<< "ms" << endl;

        out_queue.push(fut);
        if(out_queue.size() <= 1){
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
    string param_path = "../model.param";
    string model_path = "../model.bin";

    auto infer = YoloNCNN::create_infer(param_path, model_path, 0.5, 0.45); 

    if (infer == nullptr){
        printf("Infer is nullptr.\n");
        return ;
    }

    string img_path = "../img/000026.jpg";
    cv::Mat img = cv::imread(img_path);

    // warmup
    for(int i=0;i<10;i++){
        auto res = infer->commit(img).get();     // 将任务提交给推理器（推理器执行commit)
    }

    int count = 10;
    auto start = std::chrono::system_clock::now();
    for(int i=0;i<count;i++){
        auto res = infer->commit(img).get();     // 将任务提交给推理器（推理器执行commit)
    }
    
    auto end = std::chrono::system_clock::now();
    float cost_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << cost_time / (count * 1.0) << "ms" << endl;

    auto res = infer->commit(img).get();
    printBox(res);
    // for (size_t i = 0; i < res.size(); i++){
    //     auto box = res[i];
    //     cout << box.score << endl;
    //     // cv::rectangle(img, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2),
    //             //   cv::Scalar(0, 0, 255), 2);
    //     // cv::putText(img, pred->class_names[box.category], cv::Point(box.x1,
    //     // box.y1),
    //     //             cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 2);
    // }
    // cv::imwrite("result_test.jpg", img);
    return ;

}

int main(){
    t1();
    // t2();

}

