# RT-DETR的tensorrt转换

百度家的这个新模型是真不错，尤其是出了r18的可以类比yolo系列的s模型了。

## paddle infer
```shell
python tools/infer.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
              -o weights=0zzx/rtdetr_r18vd_dec3_6x_coco.pdparams \
              --infer_img=./demo/000000570688.jpg
```


## paddle onnx
paddlepaddle-gpu需要大于2.4.1要不报错。
首先需要先导出
```shell
python tools/export_model.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
              -o weights=rtdetr_r18vd_6x_coco.pdparams trt=True \
              --output_dir=output_inference
```
然后转成onnx
```shell
paddle2onnx --model_dir=rtdetr_r18vd_6x_coco \
            --model_filename model.pdmodel  \
            --params_filename model.pdiparams \
            --opset_version 16 \
            --save_file rtdetr_r18vd_6x_coco.onnx
```

## trt转换
```shell
trtexec --onnx=./rtdetr_r18vd_6x_coco.onnx \
        --workspace=4096 \
        --shapes=image:1x3x640x640 \
        --saveEngine=rtdetr_r18vd_6x_coco.trt \
        --avgRuns=100 \
        --fp16
```