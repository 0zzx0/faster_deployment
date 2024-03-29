#include "common.h"

namespace FasterNCNN{

float InterSectionArea(const ObjBox &a, const ObjBox &b){
    if(a.x1 > b.x2 || a.x2 < b.x1 || a.y1 > b.y2 || a.y2 < b.y1){
        return 0.0f;
    }
    float inter_w = std::min(a.x2, b.x2) - std::min(a.x1, b.x1);
    float inter_h = std::min(a.y2, b.y2) - std::min(a.y1, b.y1);

    return inter_w * inter_h; 
}

bool ScoreSort(ObjBox a, ObjBox b){
    return (a.score > b.score);
}

void nms(std::vector<ObjBox> &src_boxes, std::vector<ObjBox> &dst_boxes, float threshold){
    std::vector<int> picked;
    std::sort(src_boxes.begin(), src_boxes.end(), ScoreSort);

    for (int i = 0; i < src_boxes.size(); i++){
        int keep = 1;
        for(int j=0; j < picked.size(); j++){
            float inter_area = InterSectionArea(src_boxes[i], src_boxes[picked[j]]);
            float union_area = src_boxes[i].area() + src_boxes[picked[j]].area() - inter_area;
            float iou = inter_area / union_area;
            if((iou > threshold) && (src_boxes[i].category == src_boxes[picked[j]].category)){
                keep = 0;
                break;
            }
        }
        if(keep){
            picked.push_back(i);
        }
    }
    for(int i=0;i<picked.size();i++){
        dst_boxes.push_back(src_boxes[picked[i]]);
    }
    return ;

}



postProcess::postProcess(postProcessType type, float input_h, float input_w, float conf_thr, float nms_thr) {
    if(type != postProcessType::yolox) {
        INFOE("目前只支持yolox后处理! ");
        return ;
    }
    input_h_ = input_h;
    input_w_ = input_w;
    conf_thr_ = conf_thr;
    nms_thr_ = nms_thr;
    yolox_generate_grids_and_stride();

}

void postProcess::forward(ncnn::Mat &output_ ){
    out_boxes.clear();
    nms_boxes.clear();
    yolox_decode(output_);
    nms(out_boxes, nms_boxes, nms_thr_);
}

void postProcess::yolox_generate_grids_and_stride(){
    for( auto stride: strides){
        int num_grid = input_w_ / stride;
        for(int g1=0; g1<num_grid; g1++){
            for (int g0 = 0; g0<num_grid; g0++){
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}

void postProcess::yolox_decode(ncnn::Mat &output_){
    
    const int num_grid = output_.h;
    const int num_class = output_.w - 5;
    const int num_anchors = grid_strides.size();

    const float* feat_ptr = output_.channel(0);
    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++){

        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;

        // yolox/models/yolo_head.py decode logic
        //  outputs[..., :2] = (outputs[..., :2] + grids) * strides
        //  outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        float x_center = (feat_ptr[0] + grid0) * stride;
        float y_center = (feat_ptr[1] + grid1) * stride;
        float w = exp(feat_ptr[2]) * stride;
        float h = exp(feat_ptr[3]) * stride;
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;
        float x1 = x_center + w * 0.5f;
        float y1 = y_center + h * 0.5f;

        float box_objectness = feat_ptr[4];
        for (int class_idx = 0; class_idx < num_class; class_idx++){

            float box_cls_score = feat_ptr[5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > conf_thr_){

                ObjBox obj;
                obj.x1 = x0;
                obj.y1 = y0;
                obj.x2 = x1;
                obj.y2 = y1;

                obj.category = class_idx;
                obj.score = box_prob;

                out_boxes.push_back(obj);
            }

        } // class loop
        feat_ptr += output_.w;

    } // point anchor loop
}


}
