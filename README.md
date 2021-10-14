# YOLOv5-Lite：lighter, faster and easier to deploy   ![](https://zenodo.org/badge/DOI/10.5281/zenodo.5241425.svg)

![image](https://user-images.githubusercontent.com/82716366/135564164-3ec169c8-93a7-4ea3-b0dc-40f1059601ef.png)

Perform a series of ablation experiments on yolov5 to make it lighter (smaller Flops, lower memory, and fewer parameters) and faster (add shuffle channel, yolov5 head for channel reduce. It can infer at least 10+ FPS On the Raspberry Pi 4B when input the frame with 320×320) and is easier to deploy (removing the Focus layer and four slice operations, reducing the model quantization accuracy to an acceptable range).

## Comparison of ablation experiment results

  ID|Model | Input_size|Flops| Params | Size（M） |Map@0.5|Map@.5:0.95
 :-----:|:-----:|:-----:|:----------:|:----:|:----:|:----:|:----:|
001| yolo-fastest| 320×320|0.25G|0.35M|1.4| 24.4| -
002| nanodet-m| 320×320| 0.72G|0.95M|1.8|- |20.6
003| yolo-fastest-xl| 320×320|0.72G|0.92M|3.5| 34.3| -
004| YOLOv5-Lite<sub>s</sub><sup>ours</sup>| 320×320|1.43G |1.62M|3.3| 36.2|20.8| 
005| yolov3-tiny| 416×416| 6.96G|6.06M|23.0| 33.1|16.6
006| yolov4-tiny| 416×416| 5.62G|8.86M| 33.7|40.2|21.7
007| YOLOv5-Lite<sub>s</sub><sup>ours</sup>| 416×416|2.56G |1.62M|3.3| 41.3|24.4
008| YOLOv5-Lite<sub>c</sub><sup>ours</sup>| 640×640|8.6G |4.37M|9.2| 52.5|33.0| 
009| YOLOv5s| 640×640| 17.0G|7.3M|14.2| 55.8|35.9
010| YOLOv5-Lite<sub>g</sub><sup>ours</sup>| 640×640|15.7G |5.3M|10.9| 56.9|38.1| 

## Comparison on different platforms

Equipment|Computing backend|System|Input|Framework|v5Lite-s|v5Lite-c|v5Lite-g|YOLOv5s
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
Inter|@i5-10210U|window(x86)|640×640|openvino|-|46ms|-|131ms
Nvidia|@RTX 2080Ti|Linux(x86)|640×640|torch|-|-|15ms|14ms
Redmi K30|@Snapdragon 730G|Android(arm64)|320×320|ncnn|36ms|-|-|263ms
Raspberrypi 4B|@ARM Cortex-A72|Linux(arm64)|320×320|ncnn|97ms|-|-|371ms
Raspberrypi 4B|@ARM Cortex-A72|Linux(arm64)|320×320|mnn|88ms|-|-|356ms

* The above is a 4-thread test benchmark
* Raspberrypi 4B enable bf16s optimization，[Raspberrypi 64 Bit OS](http://downloads.raspberrypi.org/raspios_arm64/images/raspios_arm64-2020-08-24/)

## [ ·Model Zoo· ]

#### @YOLOv5-Lites:

Model|Size|Backbone|Head|Framework|Design for
:---:|:---:|:---:|:---:|:---:|:---:
[v5Lite-s.pt](https://drive.google.com/file/d/1by8_RZFHGcHB70nHSANXTPVtgDHZalPn/view?usp=sharing)|3.3m|shufflenetv2（Megvii）|v5Lites-head|Pytorch|Arm-cpu
v5Lite-s.bin<br />v5Lite-s.param|3.3m|shufflenetv2|v5Lites-head|ncnn|Arm-cpu
v5Lite-s-int8.bin<br />v5Lite-s-int8.param|1.7m|shufflenetv2|v5Lites-head|ncnn|Arm-cpu
v5Lite-s.mnn|3.3m|shufflenetv2|v5Lites-head|mnn|Arm-cpu
v5Lite-s-int4.mnn|987k|shufflenetv2|v5Lites-head|mnn|Arm-cpu

#### @YOLOv5-Litec:

Model|Size|Backbone|Head|Framework|Design for
:---:|:---:|:---:|:---:|:---:|:---:
v5Lite-c.pt|9m|PPLcnet（Baidu）|v5Litec-head|Pytorch|x86-cpu / x86-vpu
v5Lite-c.bin<br />v5Lite-c.xml|8.7m|PPLcnet|v5Litec-head|openvivo|x86-cpu / x86-vpu

#### @YOLOv5-Liteg:

Model|Size|Backbone|Head|Framework|Design for
:---:|:---:|:---:|:---:|:---:|:---:
[v5Lite-g.pt](https://drive.google.com/file/d/1epLouWuSLMMFcbEjAqtWLBPjNJXKi7sb/view?usp=sharing)|10.9m|Repvgg（Tsinghua）|v5Liteg-head|Pytorch|x86-gpu / arm-gpu / arm-npu
v5Lite-g-int8.engine|8.5m|Repvgg|v5Liteg-head|Tensorrt|x86-gpu / arm-gpu / arm-npu

## <div>How to use</div>

<details open>
<summary>Install</summary>

[**Python>=3.6.0**](https://www.python.org/) is required with all
[requirements.txt](https://github.com/ppogg/YOLOv5-Lite/master/requirements.txt) installed including
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
$ python train.py --data coco.yaml --cfg v5lite-s.yaml --weights v5lite-s.pt --batch-size 128
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
<summary>model hub</summary>

Here, the original components of YOLOv5 and the reproduced components of YOLOv5-Lite are organized and stored in the [model hub](https://github.com/ppogg/YOLOv5-Lite/tree/master/models/model_hub)：

  ![image](https://user-images.githubusercontent.com/82716366/135563400-2b6082c5-d0c2-49b8-9235-748950df30b8.png)

  Updating ...

</details>

## Android_demo 

This is a Redmi phone, the processor is Snapdragon 730G, and yolov5-lite is used for detection. The performance is as follows:

link: https://github.com/ppogg/YOLOv5-Lite/tree/master/ncnn_Android

Android_v5Lite-s: https://drive.google.com/file/d/1CtohY68N2B9XYuqFLiTp-Nd2kuFWgAUR/view?usp=sharing

Android_v5Lite-g: https://drive.google.com/file/d/1FnvkWxxP_aZwhi000xjIuhJ_OhqOUJcj/view?usp=sharing

## More detailed explanation

Detailed model link:
 [1] https://zhuanlan.zhihu.com/p/400545131
 
 [2] https://zhuanlan.zhihu.com/p/410874403

 [3] https://blog.csdn.net/weixin_45829462/article/details/119787840
 
 [4] https://zhuanlan.zhihu.com/p/420737659

## Reference

https://github.com/ultralytics/yolov5

https://github.com/megvii-model/ShuffleNet-Series

https://github.com/Tencent/ncnn
