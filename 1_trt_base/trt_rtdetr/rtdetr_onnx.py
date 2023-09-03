import time
import cv2
import numpy as np
import argparse
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm


COCO_CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
                "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                "scissors", "teddy bear", "hair drier", "toothbrush", ]

class PicoDet():
    def __init__(self,
                 model_pb_path,
                 prob_threshold=0.5):
        self.classes = COCO_CLASSES
        self.num_classes = len(self.classes)
        self.prob_threshold = prob_threshold
        self.mean = np.array(
            [103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(
            [57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.net = ort.InferenceSession(model_pb_path, so)
        inputs_name = [a.name for a in self.net.get_inputs()]
        inputs_shape = {
            k: v.shape
            for k, v in zip(inputs_name, self.net.get_inputs())
        }
        self.input_shape = inputs_shape['image'][2:]

    def _normalize(self, img):
        img = img.astype(np.float32)
        img = (img / 255.0 - self.mean / 255.0) / (self.std / 255.0)
        return img

    def resize_image(self, srcimg, keep_ratio=False):
        top, left, newh, neww = 0, 0, self.input_shape[0], self.input_shape[1]
        origin_shape = srcimg.shape[:2]
        im_scale_y = newh / float(origin_shape[0])
        im_scale_x = neww / float(origin_shape[1])
        img_shape = np.array([
            [float(self.input_shape[0]), float(self.input_shape[1])]
        ]).astype('float32')
        scale_factor = np.array([[im_scale_y, im_scale_x]]).astype('float32')

        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_shape[0], int(self.input_shape[1] /
                                                      hw_scale)
                img = cv2.resize(
                    srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_shape[1] - neww) * 0.5)
                img = cv2.copyMakeBorder(
                    img,
                    0,
                    0,
                    left,
                    self.input_shape[1] - neww - left,
                    cv2.BORDER_CONSTANT,
                    value=0)  # add border
            else:
                newh, neww = int(self.input_shape[0] *
                                 hw_scale), self.input_shape[1]
                img = cv2.resize(
                    srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_shape[0] - newh) * 0.5)
                img = cv2.copyMakeBorder(
                    img,
                    top,
                    self.input_shape[0] - newh - top,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=0)
        else:
            img = cv2.resize(
                srcimg, self.input_shape, interpolation=cv2.INTER_LINEAR)

        return img, img_shape, scale_factor

    def get_color_map_list(self, num_classes):
        color_map = num_classes * [0, 0, 0]
        for i in range(0, num_classes):
            j = 0
            lab = i
            while lab:
                color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
                color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
                color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
                j += 1
                lab >>= 3
        color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
        return color_map

    def detect(self, srcimg):
        img, im_shape, scale_factor = self.resize_image(srcimg)
        img = self._normalize(img)

        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

        inputs_dict = {
            'im_shape': im_shape,
            'image': blob,
            'scale_factor': scale_factor
        }
        inputs_name = [a.name for a in self.net.get_inputs()]
        net_inputs = {k: inputs_dict[k] for k in inputs_name}

        outs = self.net.run(None, net_inputs)

        outs = np.array(outs[0])
        expect_boxes = (outs[:, 1] > 0.5) & (outs[:, 0] > -1)
        np_boxes = outs[expect_boxes, :]

        print(np_boxes)

        # color_list = self.get_color_map_list(self.num_classes)
        # clsid2color = {}

        # for i in range(np_boxes.shape[0]):
        #     classid, conf = int(np_boxes[i, 0]), np_boxes[i, 1]
        #     xmin, ymin, xmax, ymax = int(np_boxes[i, 2]), int(np_boxes[
        #         i, 3]), int(np_boxes[i, 4]), int(np_boxes[i, 5])

        #     if classid not in clsid2color:
        #         clsid2color[classid] = color_list[classid]
        #     color = tuple(clsid2color[classid])

        #     cv2.rectangle(
        #         srcimg, (xmin, ymin), (xmax, ymax), color, thickness=2)
        #     print(self.classes[classid] + ': ' + str(round(conf, 3)))
        #     cv2.putText(
        #         srcimg,
        #         self.classes[classid] + ':' + str(round(conf, 3)), (xmin,
        #                                                             ymin - 10),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.8, (0, 255, 0),
        #         thickness=2)

        return srcimg

    def detect_folder(self, img_fold, result_path):
        img_fold = Path(img_fold)
        result_path = Path(result_path)
        result_path.mkdir(parents=True, exist_ok=True)

        img_name_list = filter(
            lambda x: str(x).endswith(".png") or str(x).endswith(".jpg"),
            img_fold.iterdir(), )
        img_name_list = list(img_name_list)
        print(f"find {len(img_name_list)} images")

        for img_path in tqdm(img_name_list):
            img = cv2.imread(str(img_path), 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            srcimg = net.detect(img)
            save_path = str(result_path / img_path.name.replace(".png", ".jpg"))
            cv2.imwrite(save_path, srcimg)


if __name__ == '__main__':

    model_path = "rtdetr_r18vd_6x_coco/rtdetr_r18vd_6x_coco.onnx"
    img_file = "../demo/000000570688.jpg"
    conf = 0.5
    net = PicoDet(model_path, conf)
    
    img = cv2.imread(img_file)
    t1 = time.perf_counter()
    for _ in range(100):
        net.detect(img)
    t2 = time.perf_counter()
    print(f"time: {(t2-t1)*1000/100.0}ms")
    

