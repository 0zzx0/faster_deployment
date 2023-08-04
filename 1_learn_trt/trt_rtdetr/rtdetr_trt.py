import cv2
import time
import numpy as np
import tensorrt as trt
from cuda import cudart

model_path = "rtdetr_r18vd_6x_coco/rtdetr_r18vd_6x_coco.trt"
img_path = "../demo/000000570688.jpg"

mean = np.array(
            [103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
std = np.array(
            [57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
def normalize(img):
    img = img.astype(np.float32)
    img = (img / 255.0 - mean / 255.0) / (std / 255.0)
    return img

def resize_image(srcimg, input_shape, keep_ratio=False):
    top, left, newh, neww = 0, 0, input_shape[0], input_shape[1]
    origin_shape = srcimg.shape[:2]
    im_scale_y = newh / float(origin_shape[0])
    im_scale_x = neww / float(origin_shape[1])
    img_shape = np.array([
        [float(input_shape[0]), float(input_shape[1])]
    ]).astype('float32')
    scale_factor = np.array([[im_scale_y, im_scale_x]]).astype('float32')

    if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
        hw_scale = srcimg.shape[0] / srcimg.shape[1]
        if hw_scale > 1:
            newh, neww = input_shape[0], int(input_shape[1] /
                                                    hw_scale)
            img = cv2.resize(
                srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
            left = int((input_shape[1] - neww) * 0.5)
            img = cv2.copyMakeBorder(
                img,
                0,
                0,
                left,
                input_shape[1] - neww - left,
                cv2.BORDER_CONSTANT,
                value=0)  # add border
        else:
            newh, neww = int(input_shape[0] *
                                hw_scale), input_shape[1]
            img = cv2.resize(
                srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
            top = int((input_shape[0] - newh) * 0.5)
            img = cv2.copyMakeBorder(
                img,
                top,
                input_shape[0] - newh - top,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=0)
    else:
        img = cv2.resize(
            srcimg, input_shape, interpolation=cv2.INTER_LINEAR)

    return img, img_shape, scale_factor




class RtdetrTrt:
    def __init__(self) -> None:
        self.logger = trt.Logger(trt.Logger.WARNING)
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
        print("============================================== ")

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

        img, img_shape, scale_factor = resize_image(origin_img, (640, 640))
        img = normalize(img)
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

        # cpu端数据
        self.bufferH[0] = np.ascontiguousarray(img_shape)
        self.bufferH[1] = np.ascontiguousarray(blob)
        self.bufferH[2] = np.ascontiguousarray(scale_factor)
                                                        

        for i in range(self.nInput, self.nIO):                # 输出数据
            self.bufferH[i] = np.empty(self.context.get_tensor_shape(self.lTensorName[i]), 
                                    dtype=trt.nptype(self.engine.get_tensor_dtype(self.lTensorName[i])))
        
        # 输入数据复制到显存
        for i in range(self.nInput):                                                    
            cudart.cudaMemcpy(self.bufferD[i], self.bufferH[i].ctypes.data, self.bufferH[i].nbytes, 
                            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        
        # # 推理 execute_async_v2 execute_v2
        self.context.execute_v2(self.bufferD)     # batchsize bingings
        for i in range(self.nInput, self.nIO):    # 数据拷会cpu
            cudart.cudaMemcpy(self.bufferH[i].ctypes.data, self.bufferD[i], self.bufferH[i].nbytes, 
                            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        # for i in range(self.nInput, self.nIO):  
        #     print(len(self.bufferH[i]))
        dets = []
        for i in self.bufferH[-1]:
            if i[1] > 0.5:
                dets.append(i)
        return dets
    

    def infer_(self, img_shape, blob, scale_factor):
        # cpu端数据
        self.bufferH[0] = np.ascontiguousarray(img_shape)
        self.bufferH[1] = np.ascontiguousarray(blob)
        self.bufferH[2] = np.ascontiguousarray(scale_factor)
                                                        

        for i in range(self.nInput, self.nIO):                # 输出数据
            self.bufferH[i] = np.empty(self.context.get_tensor_shape(self.lTensorName[i]), 
                                    dtype=trt.nptype(self.engine.get_tensor_dtype(self.lTensorName[i])))
        
        # 输入数据复制到显存
        for i in range(self.nInput):                                                    
            cudart.cudaMemcpy(self.bufferD[i], self.bufferH[i].ctypes.data, self.bufferH[i].nbytes, 
                            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        
        # # 推理 execute_async_v2 execute_v2
        self.context.execute_v2(self.bufferD)     # batchsize bingings
        for i in range(self.nInput, self.nIO):    # 数据拷会cpu
            cudart.cudaMemcpy(self.bufferH[i].ctypes.data, self.bufferD[i], self.bufferH[i].nbytes, 
                            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        # for i in range(self.nInput, self.nIO):  
        #     print(len(self.bufferH[i]))
        dets = []
        for i in self.bufferH[-1]:
            if i[1] > 0.5:
                dets.append(i)
        return dets
    

    def myfree(self):
        for i in self.bufferD:       # 释放显存  
            cudart.cudaFree(i)


if __name__ == '__main__':
    origin_img = cv2.imread(img_path)

    img, img_shape, scale_factor = resize_image(origin_img, (640, 640))
    img = normalize(img)
    blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)


    yolo_trt = RtdetrTrt()
    yolo_trt.infer(origin_img)
    for _ in range(50):
        # yolo_trt.infer(origin_img)
        yolo_trt.infer_(img, img_shape, scale_factor)

    time_b = time.perf_counter()
    for _ in range(1000):
        # dets = yolo_trt.infer(origin_img)
        dets = yolo_trt.infer_(img, img_shape, scale_factor)

    time_e = time.perf_counter()
    print(f"cost time: {(time_e-time_b)*1000 / 1000.0 :.2f}ms")

    for i in dets:
        print(f"class: {i[0]:.0f}\tsocre: {i[1] :.2f}\tx1: {i[2] :.0f}\ty1: {i[3] :.0f}\tx2: {i[4] :.0f}\ty2: {i[5] :.0f}")



    yolo_trt.myfree()

