#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../apps/yolo/yolo.h"
#include "save.hpp"

using namespace std;

const string base_path = "/home/zzx/Experiment/Data/UTDAC2020/val2017/";
YOLO::YoloType type = YOLO::YoloType::X;
const string model_file = "../yolox_b16.engine";
const int deviceid = 0;

const float confidence_threshold = 0.5f;
const float nms_threshold = 0.65f;

int main() {
    int batch_size = 1;
    YOLO::set_device(deviceid);

    auto yolo = YOLO::create_infer(model_file, type, deviceid, batch_size, confidence_threshold,
                                   nms_threshold);

    ifstream img_id("../src/eval/img_id.txt");
    vector<string> all_id;
    vector<string> all_img;

    while(!img_id.eof()) {
        string id;
        string name;
        img_id >> id;
        img_id >> name;
        if(id.size() == 0) break;

        all_id.push_back(id);
        all_img.push_back(name);
        // cout << id << " " << name << endl;
    }
    img_id.close();

    string resfile_name = "../src/eval/results.txt";
    SaveResult resfile(resfile_name);

    assert(all_id.size() == all_img.size());

    for(int i = 0; i < all_id.size(); i++) {
        string cur_img = base_path + all_img[i];
        auto image = cv::imread(cur_img);
        auto objs = yolo->commit(image);
        auto res = objs.get();
        for(auto& one : res) {
            int x = one.left;
            int y = one.top;
            int w = one.right - one.left;
            int h = one.bottom - one.top;
            vector<int> xywh{x, y, w, h};
            // cout << one.left << one.right << one.bottom << one.top << endl;
            resfile.save_one_line(all_img[i], all_id[i], one.class_label, one.confidence, xywh);
        }
    }
    return 0;
}