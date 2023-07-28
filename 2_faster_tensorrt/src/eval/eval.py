from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import json


def get_coco_from_txt(txtfile, json_file, clsid2catid):

    dataset_res = []

    with open(txtfile, 'r') as f:
        datas = f.readlines()
        # print(len(datas))

        for data in datas:
            info = data.split(" ")[:-1]
            result = {}
            result["image_id"] = int(info[1])
            result["category_id"] = clsid2catid[int(info[2])]
            result["bbox"] = [int(info[4]), int(info[5]), int(info[6]), int(info[7])]
            result["score"] = float(info[3])
            dataset_res.append(result)

    with open(json_file, "w") as f:
        json.dump(dataset_res, f)
    print("json 保存成功")


annFile = "/home/zzx/Experiment/Data/UTDAC2020/annotations/instances_val2017.json"
resFile = "./results.txt"
resJson = 'eval_results.json'

cocoGt=COCO(annFile)
clsid2catid = cocoGt.getCatIds()

get_coco_from_txt(resFile, resJson, clsid2catid)
cocoDt = cocoGt.loadRes(resJson)

cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
cocoEval.params.imgIds = cocoGt.getImgIds()
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()


