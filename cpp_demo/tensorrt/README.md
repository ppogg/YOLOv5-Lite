# YOLOv5 Lite PyTorch=>ONNX=>TensorRT

## 1.Export trt Model

```python
python export.py ---weights weights/v5lite-g.pt --batch-size 1 --imgsz 640 --include onnx --simplify

trtexec --explicitBatch --onnx=./v5lite-g.onnx --saveEngine=v5lite-g.trt --fp16
```

## 2.Build yolov5 TensorRT Inference Project

```
mkdir build && cd build
cmake ..
make -j
```

## 3.Run yolov5_trt

- inference dir with v5lite-g
```
./yolov5_trt ../config.yaml ../samples
```
## 4.Results:

![](E:\星球\yolov5-tensorrt\samples\person_.jpg)
