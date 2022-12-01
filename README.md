# YOLOv5-Lite：Lighter, faster and easier to deploy   ![](https://zenodo.org/badge/DOI/10.5281/zenodo.5241425.svg)

![论文插图](https://user-images.githubusercontent.com/82716366/167448925-a431d3a4-ad5d-491d-be95-c90701122a54.png)

Perform a series of ablation experiments on yolov5 to make it lighter (smaller Flops, lower memory, and fewer parameters) and faster (add shuffle channel, yolov5 head for channel reduce. It can infer at least 10+ FPS On the Raspberry Pi 4B when input the frame with 320×320) and is easier to deploy (removing the Focus layer and four slice operations, reducing the model quantization accuracy to an acceptable range).

![image](https://user-images.githubusercontent.com/82716366/135564164-3ec169c8-93a7-4ea3-b0dc-40f1059601ef.png)

## Comparison of ablation experiment results

  ID|Model | Input_size|Flops| Params | Size（M） |Map@0.5|Map@.5:0.95
 :-----:|:-----:|:-----:|:----------:|:----:|:----:|:----:|:----:|
001| yolo-fastest| 320×320|0.25G|0.35M|1.4| 24.4| -
002| YOLOv5-Lite<sub>e</sub><sup>ours</sup>|320×320|0.73G|0.78M|1.7| 35.1|-|
003| NanoDet-m| 320×320| 0.72G|0.95M|1.8|- |20.6
004| yolo-fastest-xl| 320×320|0.72G|0.92M|3.5| 34.3| -
005| YOLOX<sub>Nano</sub>|416×416|1.08G|0.91M|7.3(fp32)| -|25.8|
006| yolov3-tiny| 416×416| 6.96G|6.06M|23.0| 33.1|16.6
007| yolov4-tiny| 416×416| 5.62G|8.86M| 33.7|40.2|21.7
008| YOLOv5-Lite<sub>s</sub><sup>ours</sup>| 416×416|1.66G |1.64M|3.4| 42.0|25.2
009| YOLOv5-Lite<sub>c</sub><sup>ours</sup>| 512×512|5.92G |4.57M|9.2| 50.9|32.5| 
010| NanoDet-EfficientLite2| 512×512| 7.12G|4.71M|18.3|- |32.6
011| YOLOv5s(6.0)| 640×640| 16.5G|7.23M|14.0| 56.0|37.2
012| YOLOv5-Lite<sub>g</sub><sup>ours</sup>| 640×640|15.6G |5.39M|10.9| 57.6|39.1| 

See the wiki: https://github.com/ppogg/YOLOv5-Lite/wiki/Test-the-map-of-models-about-coco

## Comparison on different platforms

Equipment|Computing backend|System|Input|Framework|v5lite-e|v5lite-s|v5lite-c|v5lite-g|YOLOv5s
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
Inter|@i5-10210U|window(x86)|640×640|openvino|-|-|46ms|-|131ms
Nvidia|@RTX 2080Ti|Linux(x86)|640×640|torch|-|-|-|15ms|14ms
Redmi K30|@Snapdragon 730G|Android(armv8)|320×320|ncnn|27ms|38ms|-|-|163ms
Xiaomi 10|@Snapdragon 865|Android(armv8)|320×320|ncnn|10ms|14ms|-|-|163ms
Raspberrypi 4B|@ARM Cortex-A72|Linux(arm64)|320×320|ncnn|-|84ms|-|-|371ms
Raspberrypi 4B|@ARM Cortex-A72|Linux(arm64)|320×320|mnn|-|76ms|-|-|356ms
AXera-Pi|Cortex A7@CPU<br />3.6TOPs @NPU|Linux(arm64)|640×640|axpi|-|-|-|22ms|22ms


* The above is a 4-thread test benchmark
* Raspberrypi 4B enable bf16s optimization，[Raspberrypi 64 Bit OS](http://downloads.raspberrypi.org/raspios_arm64/images/raspios_arm64-2020-08-24/)

###  qq交流群：993965802

入群答案:剪枝 or 蒸馏 or 量化 or 低秩分解（任意其一均可）

##  ·Model Zoo· 

#### @v5lite-e:

Model|Size|Backbone|Head|Framework|Design for
:---:|:---:|:---:|:---:|:---:|:---
v5Lite-e.pt|1.7m|shufflenetv2（Megvii）|v5Litee-head|Pytorch|Arm-cpu
v5Lite-e.bin<br />v5Lite-e.param|1.7m|shufflenetv2|v5Litee-head|ncnn|Arm-cpu
v5Lite-e-int8.bin<br />v5Lite-e-int8.param|0.9m|shufflenetv2|v5Litee-head|ncnn|Arm-cpu
v5Lite-e-fp32.mnn|3.0m|shufflenetv2|v5Litee-head|mnn|Arm-cpu
v5Lite-e-fp32.tnnmodel<br />v5Lite-e-fp32.tnnproto|2.9m|shufflenetv2|v5Litee-head|tnn|arm-cpu
v5Lite-e-320.onnx|3.1m|shufflenetv2|v5Litee-head|onnxruntime|x86-cpu

#### @v5lite-s:

Model|Size|Backbone|Head|Framework|Design for
:---:|:---:|:---:|:---:|:---:|:---
v5Lite-s.pt|3.4m|shufflenetv2（Megvii）|v5Lites-head|Pytorch|Arm-cpu
v5Lite-s.bin<br />v5Lite-s.param|3.3m|shufflenetv2|v5Lites-head|ncnn|Arm-cpu
v5Lite-s-int8.bin<br />v5Lite-s-int8.param|1.7m|shufflenetv2|v5Lites-head|ncnn|Arm-cpu
v5Lite-s.mnn|3.3m|shufflenetv2|v5Lites-head|mnn|Arm-cpu
v5Lite-s-int4.mnn|987k|shufflenetv2|v5Lites-head|mnn|Arm-cpu
v5Lite-s-fp16.bin<br />v5Lite-s-fp16.xml|3.4m|shufflenetv2|v5Lites-head|openvivo|x86-cpu
v5Lite-s-fp32.bin<br />v5Lite-s-fp32.xml|6.8m|shufflenetv2|v5Lites-head|openvivo|x86-cpu
v5Lite-s-fp16.tflite|3.3m|shufflenetv2|v5Lites-head|tflite|arm-cpu
v5Lite-s-fp32.tflite|6.7m|shufflenetv2|v5Lites-head|tflite|arm-cpu
v5Lite-s-int8.tflite|1.8m|shufflenetv2|v5Lites-head|tflite|arm-cpu
v5Lite-s-416.onnx|6.4m|shufflenetv2|v5Lites-head|onnxruntime|x86-cpu

#### @v5lite-c:

Model|Size|Backbone|Head|Framework|Design for
:---:|:---:|:---:|:---:|:---:|:---:
v5Lite-c.pt|9m|PPLcnet（Baidu）|v5s-head|Pytorch|x86-cpu / x86-vpu
v5Lite-c.bin<br />v5Lite-c.xml|8.7m|PPLcnet|v5s-head|openvivo|x86-cpu / x86-vpu
v5Lite-c-512.onnx|18m|PPLcnet|v5s-head|onnxruntime|x86-cpu

#### @v5lite-g:

Model|Size|Backbone|Head|Framework|Design for
:---:|:---:|:---:|:---:|:---:|:---:
v5Lite-g.pt|10.9m|Repvgg（Tsinghua）|v5Liteg-head|Pytorch|x86-gpu / arm-gpu / arm-npu
v5Lite-g-int8.engine|8.5m|Repvgg-yolov5|v5Liteg-head|Tensorrt|x86-gpu / arm-gpu / arm-npu
v5lite-g-int8.tmfile|8.7m|Repvgg-yolov5|v5Liteg-head|Tengine| arm-npu
v5Lite-g-640.onnx|21m|Repvgg-yolov5|yolov5-head|onnxruntime|x86-cpu
v5Lite-g-640.joint|7.1m|Repvgg-yolov5|yolov5-head|axpi|arm-npu

#### Download Link：

> - [ ] `v5lite-e.pt`:   | [Baidu Drive](https://pan.baidu.com/s/1bjXo7KIFkOnB3pxixHeMPQ)  | [Google Drive](https://drive.google.com/file/d/1_DvT_qjznuE-ev_pDdGKwRV3MjZ3Zos8/view?usp=sharing) |<br> 
>> |──────`ncnn-fp16`:   | [Baidu Drive](https://pan.baidu.com/s/1_QvWvkhHB7kdcRZ6k4at1g)  | [Google Drive](https://drive.google.com/drive/folders/1w4mThJmqjhT1deIXMQAQ5xjWI3JNyzUl?usp=sharing) |<br> 
>> |──────`ncnn-int8`: | [Baidu Drive](https://pan.baidu.com/s/1JO8qbbVx6zJ-6aq5EgM6PA) | [Google Drive](https://drive.google.com/drive/folders/1YNtNVWlRqN8Dwc_9AtRkN0LFkDeJ92gN?usp=sharing) |<br> 
>> └──────`onnx-fp32`: | [Baidu Drive](https://pan.baidu.com/s/1gwLqiPLTDjlSqWJ7AnEB1A) | [Google Drive](https://drive.google.com/file/d/15_z6VlbWuonsHak-7bdtw-QOcvOaMddB/view?usp=sharing) |<br> 
> - [ ] `v5lite-s.pt`:   | [Baidu Drive](https://pan.baidu.com/s/1j0n0K1kqfv1Ouwa2QSnzCQ)  | [Google Drive](https://drive.google.com/file/d/1ccLTmGB5AkKPjDOyxF3tW7JxGWemph9f/view?usp=sharing) |<br> 
>> |──────`ncnn-fp16`:   | [Baidu Drive](https://pan.baidu.com/s/1kWtwx1C0OTTxbwqJyIyXWg)  | [Google Drive](https://drive.google.com/drive/folders/1w4mThJmqjhT1deIXMQAQ5xjWI3JNyzUl?usp=sharing) |<br> 
>> |──────`ncnn-int8`: | [Baidu Drive](https://pan.baidu.com/s/1QX6-oNynrW-f3i0P0Hqe4w) | [Google Drive](https://drive.google.com/drive/folders/1YNtNVWlRqN8Dwc_9AtRkN0LFkDeJ92gN?usp=sharing) |<br> 
>> |──────`mnn-fp16`: | [Baidu Drive](https://pan.baidu.com/s/12lOtPTl4xujWm5BbFJh3zA) | [Google Drive](https://drive.google.com/drive/folders/1PpFoZ4b8mVs1GmMxgf0WUtXUWaGK_JZe?usp=sharing) |<br> 
>> |──────`mnn-int4`: | [Baidu Drive](https://pan.baidu.com/s/11fbjFi18xkq4ltAKUKDOCA) | [Google Drive](https://drive.google.com/drive/folders/1mSU8g94c77KKsHC-07p5V3tJOZYPQ-g6?usp=sharing) |<br> 
>> |──────`onnx-fp32`: | [Baidu Drive](https://pan.baidu.com/s/1gwLqiPLTDjlSqWJ7AnEB1A) | [Google Drive](https://drive.google.com/file/d/123feVchyuqCRZV038I1Gn1gpJEVK4GFh/view?usp=sharing) |<br> 
>> └──────`tengine-fp32`: | [Baidu Drive](https://pan.baidu.com/s/123r630O8Fco7X59wFU1crA) | [Google Drive](https://drive.google.com/drive/folders/1VWmI2BC9MjH7BsrOz4VlSDVnZMXaxGOE?usp=sharing) |<br>                
> - [ ] `v5lite-c.pt`: [Baidu Drive](https://pan.baidu.com/s/1obs6uRB79m8e3uASVR6P1A) | [Google Drive](https://drive.google.com/file/d/1lHYRQKjqKCRXghUjwWkUB0HQ8ccKH6qa/view?usp=sharing) |<br> 
>> |──────`onnx-fp32`: | [Baidu Drive](https://pan.baidu.com/s/1gwLqiPLTDjlSqWJ7AnEB1A) | [Google Drive](https://drive.google.com/file/d/1VJBfZPikTce5vUatC2ZsAWQlmMdcArs2/view?usp=sharing) |<br> 
>> └──────`openvino-fp16`: | [Baidu Drive](https://pan.baidu.com/s/18p8HAyGJdmo2hham250b4A) | [Google Drive](https://drive.google.com/drive/folders/1s4KPSC4B0shG0INmQ6kZuPLnlUKAATyv?usp=sharing) |<br> 
> - [ ] `v5lite-g.pt`: | [Baidu Drive](https://pan.baidu.com/s/14zdTiTMI_9yTBgKGbv9pQw) | [Google Drive](https://drive.google.com/file/d/1oftzqOREGqDCerf7DtD5BZp9YWELlkMe/view?usp=sharing) |<br> 
>> |──────`onnx-fp32`: | [Baidu Drive](https://pan.baidu.com/s/1gwLqiPLTDjlSqWJ7AnEB1A) | [Google Drive](https://drive.google.com/file/d/1bJByk9eoS6pv8Z3N4bcLRCV3i7uk24aU/view?usp=sharing) |<br> 
>> └──────`axpi-int8`: [Google Drive](https://github.com/AXERA-TECH/ax-models/blob/main/ax620/v5Lite-g-sim-640.joint) |<br> 



Baidu Drive Password: `pogg`

#### v5lite-s model: TFLite Float32, Float16, INT8, Dynamic range quantization, ONNX, TFJS, TensorRT, OpenVINO IR FP32/FP16, Myriad Inference Engin Blob, CoreML
[https://github.com/PINTO0309/PINTO_model_zoo/tree/main/180_YOLOv5-Lite](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/180_YOLOv5-Lite)

#### Thanks for PINTO0309:[https://github.com/PINTO0309/PINTO_model_zoo/tree/main/180_YOLOv5-Lite](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/180_YOLOv5-Lite)


## <div>How to use</div>

<details open>
<summary>Install</summary>

[**Python>=3.6.0**](https://www.python.org/) is required with all
[requirements.txt](https://github.com/ppogg/YOLOv5-Lite/blob/master/requirements.txt) installed including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/):
<!-- $ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev -->

```bash
$ git clone https://github.com/ppogg/YOLOv5-Lite
$ cd YOLOv5-Lite
$ pip install -r requirements.txt
```

</details>

<details>
<summary>Inference with detect.py</summary>

`detect.py` runs inference on a variety of sources, downloading models automatically from
the [latest YOLOv5-Lite release](https://github.com/ppogg/YOLOv5-Lite/releases) and saving results to `runs/detect`.

```bash
$ python detect.py --source 0  # webcam
                            file.jpg  # image 
                            file.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            'https://youtu.be/NUsoVlDFqZg'  # YouTube
                            'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>

<details open>
<summary>Training</summary>

```bash
$ python train.py --data coco.yaml --cfg v5lite-e.yaml --weights v5lite-e.pt --batch-size 128
                                         v5lite-s.yaml           v5lite-s.pt              128
                                         v5lite-c.yaml           v5lite-c.pt               96
                                         v5lite-g.yaml           v5lite-g.pt               64
```

 If you use multi-gpu. It's faster several times:
  
 ```bash
$ python -m torch.distributed.launch --nproc_per_node 2 train.py
```
  
</details>  

</details>

<details open>
<summary>DataSet</summary>

Training set and test set distribution （the path with xx.jpg）
  
 ```bash
train: ../coco/images/train2017/
val: ../coco/images/val2017/
```
```bash
├── images            # xx.jpg example
│   ├── train2017        
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── 000003.jpg
│   └── val2017         
│       ├── 100001.jpg
│       ├── 100002.jpg
│       └── 100003.jpg
└── labels             # xx.txt example      
    ├── train2017       
    │   ├── 000001.txt
    │   ├── 000002.txt
    │   └── 000003.txt
    └── val2017         
        ├── 100001.txt
        ├── 100002.txt
        └── 100003.txt
```
  
</details> 

<details open>
<summary>Auto LabelImg</summary>

[**Link** ：https://github.com/ppogg/AutoLabelImg](https://github.com/ppogg/AutoLabelImg)  

You can use LabelImg based YOLOv5-5.0 and YOLOv5-Lite to AutoAnnotate, biubiubiu 🚀 🚀 🚀 
<img src="https://user-images.githubusercontent.com/82716366/177030174-dc3a5827-2821-4d8c-8d78-babe83c42fbf.JPG" width="950"/><br/>

  
</details> 

<details open>
<summary>Model Hub</summary>

Here, the original components of YOLOv5 and the reproduced components of YOLOv5-Lite are organized and stored in the [model hub](https://github.com/ppogg/YOLOv5-Lite/tree/master/models/hub)：

  ![modelhub](https://user-images.githubusercontent.com/82716366/146787562-e2c1c4c1-726e-4efc-9eae-d92f34333e8d.jpg)
  
  <details open>
<summary>Heatmap Analysis</summary>

  
   ```bash
$ python main.py --type all
```
  
![论文插图2](https://user-images.githubusercontent.com/82716366/167449474-3689c2bf-197a-4403-849c-b85db6bcc476.png)

  Updating ...

</details>

## How to deploy

[**ncnn**](https://github.com/ppogg/YOLOv5-Lite/blob/master/cpp_demo/ncnn/README.md)  for arm-cpu

[**mnn**](https://github.com/ppogg/YOLOv5-Lite/blob/master/cpp_demo/mnn/README.md) for arm-cpu

[**openvino**](https://github.com/ppogg/YOLOv5-Lite/blob/master/python_demo/openvino/README.md) x86-cpu or x86-vpu 

[**tensorrt(C++)**](https://github.com/ppogg/YOLOv5-Lite/blob/master/cpp_demo/tensorrt/README.md) for arm-gpu or arm-npu or x86-gpu

[**tensorrt(Python)**](https://github.com/ppogg/YOLOv5-Lite/tree/master/python_demo/tensorrt) for arm-gpu or arm-npu or x86-gpu

[**Android**](https://github.com/ppogg/YOLOv5-Lite/blob/master/android_demo/ncnn-android-v5lite/README.md) for arm-cpu

## Android_demo 

This is a Redmi phone, the processor is Snapdragon 730G, and yolov5-lite is used for detection. The performance is as follows:

link: https://github.com/ppogg/YOLOv5-Lite/tree/master/android_demo/ncnn-android-v5lite

Android_v5Lite-s: https://drive.google.com/file/d/1CtohY68N2B9XYuqFLiTp-Nd2kuFWgAUR/view?usp=sharing

Android_v5Lite-g: https://drive.google.com/file/d/1FnvkWxxP_aZwhi000xjIuhJ_OhqOUJcj/view?usp=sharing

new android app:[link] https://pan.baidu.com/s/1PRhW4fI1jq8VboPyishcIQ [keyword] pogg

<img src="https://user-images.githubusercontent.com/82716366/149959014-5f027b1c-67b6-47e2-976b-59a7c631b0f2.jpg" width="650"/><br/>

## More detailed explanation

#### Detailed model link:
 
 What is YOLOv5-Lite S/E model:
 zhihu link (Chinese): https://zhuanlan.zhihu.com/p/400545131
 
 What is YOLOv5-Lite C model:
 zhihu link (Chinese): https://zhuanlan.zhihu.com/p/420737659

 What is YOLOv5-Lite G model:
 zhihu link (Chinese): https://zhuanlan.zhihu.com/p/410874403
 
 How to deploy on ncnn with fp16 or int8:
 csdn link (Chinese): https://blog.csdn.net/weixin_45829462/article/details/119787840
 
 How to deploy on onnxruntime:
 zhihu link (Chinese): https://zhuanlan.zhihu.com/p/476533259
 
 How to deploy on tensorrt:
 zhihu link (Chinese): https://zhuanlan.zhihu.com/p/478630138
 
 How to optimize on tensorrt:
 zhihu link (Chinese): https://zhuanlan.zhihu.com/p/463074494

## Reference

https://github.com/ultralytics/yolov5

https://github.com/megvii-model/ShuffleNet-Series

https://github.com/Tencent/ncnn

## Citing YOLOv5-Lite
If you use YOLOv5-Lite in your research, please cite our work and give a star ⭐:

```
 @misc{yolov5lite2021,
  title = {YOLOv5-Lite: Lighter, faster and easier to deploy},
  author = {Xiangrong Chen and Ziman Gong},
  doi = {10.5281/zenodo.5241425}
  year={2021}
}
```
