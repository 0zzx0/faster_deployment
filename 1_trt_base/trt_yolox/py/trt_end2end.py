from typing import Tuple
import cv2
import time
import numpy as np
import tensorrt as trt
from cuda import cudart


model_path = "/home/zzx/Github/zzx_yolo/EXTRA_PKG/TensorRT-8.5.3.1/bin/yolox_end2end.engine"
img_path = "../../imgs/000026.jpg"
score_thr = 0.5

COCO_CLASSES = ('echinus', 'starfish', 'holothurian', 'scallop')
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
    ]
).astype(np.float32).reshape(-1, 3)


def preproc(img: np.ndarray, input_size: tuple, swap: tuple=(2, 0, 1))->Tuple[np.ndarray, float]:
    padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(img, 
                             (int(img.shape[1] * r), int(img.shape[0] * r)),
                             interpolation=cv2.INTER_LINEAR,
                             ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img



class YoloTRT:
    def __init__(self) -> None:
        # 构建阶段
        self.logger = trt.Logger(trt.Logger.WARNING)     # logger
        trt.init_libnvinfer_plugins( self.logger, namespace='')   # 加载插件

        self.runtime = trt.Runtime(self.logger)

        with open(model_path, "rb") as f:
            serialized_engine = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(serialized_engine)
        self.context = self.engine.create_execution_context()

        self.nIO = self.engine.num_io_tensors     # io变量数量
        self.lTensorName = [self.engine.get_tensor_name(i) for i in range(self.nIO)]   # 获取io变量名字
        self.nInput = [self.engine.get_tensor_mode(self.lTensorName[i]) for i in range(self.nIO)].count(trt.TensorIOMode.INPUT) # 输入tensor数量
        self.Output = [self.engine.get_tensor_mode(self.lTensorName[i]) for i in range(self.nIO)].count(trt.TensorIOMode.OUTPUT)
        
        print("===============INPUT/OUTPUT=================== ")
        for i in range(self.nIO):
            print(f"[{i}]{'Input ' if i < self.nInput else 'Output'} -> "+
                f"{self.engine.get_tensor_dtype(self.lTensorName[i])} " +       # 数据类型
                f"{self.engine.get_tensor_shape(self.lTensorName[i])} " +       # engine形状
                f"{self.context.get_tensor_shape(self.lTensorName[i])} " +      # context形状
                f"{self.lTensorName[i]} ")                                 # 名字
        print("============================================ ")

        # cpu端数据
        self.bufferH = []                                                            
        for i in range(self.nIO):
            self.bufferH.append(np.empty(self.context.get_tensor_shape(self.lTensorName[i]), 
                                         dtype=trt.nptype(self.engine.get_tensor_dtype(self.lTensorName[i]))))
    
        # # gpu端数据申请显存
        self.bufferD = []
        for i in range(self.nIO):
            self.bufferD.append(cudart.cudaMalloc(self.bufferH[i].nbytes)[1])
    
    def infer(self, origin_img):
        data, ratio = preproc(origin_img, (640, 640))
        # cpu端数据
        self.bufferH = []    
        self.bufferH.append(data)  # 输入数据转内存连续                                                        
        # self.bufferH.append(np.ascontiguousarray(data))  # 输入数据转内存连续

        for i in range(self.nInput, self.nIO):                # 输出数据
            self.bufferH.append(np.empty(self.context.get_tensor_shape(self.lTensorName[i]), 
                                    dtype=trt.nptype(self.engine.get_tensor_dtype(self.lTensorName[i]))))
        
        # 输入数据复制到显存
        for i in range(self.nInput):                                                    
            cudart.cudaMemcpy(self.bufferD[i], self.bufferH[i].ctypes.data, self.bufferH[i].nbytes, 
                            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        
        # # 推理
        self.context.execute_v2(self.bufferD)     # batchsize bingings
        for i in range(self.nInput, self.nIO):    # 数据拷会cpu
            cudart.cudaMemcpy(self.bufferH[i].ctypes.data, self.bufferD[i], self.bufferH[i].nbytes, 
                            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        
        dets = self.bufferH[self.nInput:self.nIO]
    
        return dets
    
    # def plot_save(self, origin_img, dets):
    #     final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
    #     origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
    #                     conf=score_thr, class_names=COCO_CLASSES)
    #     cv2.imwrite("ans.jpg", origin_img)

    def myfree(self):
        for i in self.bufferD:       # 释放显存  
            cudart.cudaFree(i)



if __name__ == '__main__':
    origin_img = cv2.imread(img_path)

    yolo_trt = YoloTRT()
    for _ in range(50):
        yolo_trt.infer(origin_img)

    time_b = time.perf_counter()
    for _ in range(1000):
        dets = yolo_trt.infer(origin_img)

    time_e = time.perf_counter()
    print(f"cost time: {(time_e-time_b)*1000 / 1000.0 :.2f}ms")
    # print(dets)

    # yolo_trt.plot_save(origin_img, dets)

    yolo_trt.myfree()
