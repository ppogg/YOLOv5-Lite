## 使用MNN部署YoloV5模型  

### 一. 将 ONNX 模型转换为 MNN 模型  

执行命令：  
```
./MNN-1.1.0/build/MNNConvert -f ONNX --modelFile yolov5-sort-cpp/yolov5ss.onnx --MNNModel yolov5ss.mnn --bizCode MNN
```
转换成功，输出以下信息：  
```
MNNConverter Version: 0.2.1.5git - MNN @ 2018

Start to Convert Other Model Format To MNN Model...
[16:42:51] /media/lihongjie/Windows/work/code/MNN-1.1.0/tools/converter/source/onnx/onnxConverter.cpp:31: ONNX Model ir version: 6
Start to Optimize the MNN Net...
[16:42:51] /media/lihongjie/Windows/work/code/MNN-1.1.0/tools/converter/source/optimizer/PostConverter.cpp:64: Inputs: images
[16:42:51] /media/lihongjie/Windows/work/code/MNN-1.1.0/tools/converter/source/optimizer/PostConverter.cpp:69: Outputs: 415
[16:42:51] /media/lihongjie/Windows/work/code/MNN-1.1.0/tools/converter/source/optimizer/PostConverter.cpp:69: Outputs: output
[16:42:51] /media/lihongjie/Windows/work/code/MNN-1.1.0/tools/converter/source/optimizer/PostConverter.cpp:69: Outputs: 395
Converted Done!
```