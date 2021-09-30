# YOLOv5-Lite：lighter, faster and easier to deploy   ![](https://zenodo.org/badge/DOI/10.5281/zenodo.5241425.svg)

![01111](https://user-images.githubusercontent.com/82716366/135464047-9a66eb1a-38d6-4585-aacb-552c70b18457.jpg)

Perform a series of ablation experiments on yolov5 to make it lighter (smaller Flops, lower memory, and fewer parameters) and faster (add shuffle channel, yolov5 head for channel reduce. It can infer at least 10+ FPS On the Raspberry Pi 4B when input the frame with 320×320) and is easier to deploy (removing the Focus layer and four slice operations, reducing the model quantization accuracy to an acceptable range).

## Comparison of ablation experiment results

  ID|Model | Input_size|Flops| Params | Size（M） |Map@0.5|Map@.5:0.95
 :-----:|:-----:|:-----:|:----------:|:----:|:----:|:----:|:----:|
001| yolo-fastest| 320×320|0.25G|0.35M|1.4| 24.4| -
002| nanodet-m| 320×320| 0.72G|0.95M|1.8|- |20.6
003| yolo-fastest-xl| 320×320|0.72G|0.92M|3.5| 34.3| -
004| YOLOv5-Lite-s| 320×320|1.43G |1.62M|3.3| 36.2|20.8| 
005| yolov3-tiny| 416×416| 6.96G|6.06M|23.0| 33.1|16.6
006| yolov4-tiny| 416×416| 5.62G|8.86M| 33.7|40.2|21.7
007| nanodet-m| 416×416| 1.2G	|0.95M|1.8|- |23.5
008| YOLOv5-Lite-s| 416×416|2.42G |1.62M|3.3| 41.3|24.4| 
009| YOLOv5-Lite-g| 416×416|7.3G |5.3M|10.9| 53.1|34.7| 
010| yolov5s| 640×640| 17.0G|7.3M|14.2| 55.8|35.9
011| YOLOv5-Lite-g| 640×640|15.7G |5.3M|10.9| 56.9|38.1| 

## Comparison on different platforms

Equipment|Computing backend|System|Framework|Input|[v5Lite-s](https://drive.google.com/file/d/1by8_RZFHGcHB70nHSANXTPVtgDHZalPn/view?usp=sharing)|[v5Lite-g](https://drive.google.com/file/d/1epLouWuSLMMFcbEjAqtWLBPjNJXKi7sb/view?usp=sharing)|YOLOv5s
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
Inter|@i5-10210U|window(x86)|640×640|torch-cpu|112ms|183ms|179ms
Nvidia|@RTX 2080Ti|Linux(x86)|640×640|torch-gpu|11ms|16ms|14ms
Redmi K30|@Snapdragon 730G|Android(arm64)|320×320|ncnn|36ms|-|-
Raspberrypi 4B|@ARM Cortex-A72|Linux(arm64)|320×320|ncnn|97ms|415ms|371ms
Raspberrypi 4B|@ARM Cortex-A72|Linux(arm64)|320×320|mnn|88ms|393ms|356ms

* The above is a 4-thread test benchmark
* Raspberrypi 4B enable bf16s optimization，[Raspberrypi 64 Bit OS](http://downloads.raspberrypi.org/raspios_arm64/images/raspios_arm64-2020-08-24/)

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
                                         v5lite-g.yaml           v5lite-g.pt               64
```

 If you use multi-gpu. It's faster several times:
  
 ```bash
$ python -m torch.distributed.launch --nproc_per_node 4 train.py
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
├── images
│   ├── train2017        # TrainSet example
│   │   ├── 000050.jpg
│   │   ├── 000051.jpg
│   │   └── 000052.jpg
│   └── val2017          # ValSet example
│       ├── 001800.jpg
│       ├── 001801.jpg
│       └── 001802.jpg
└── labels               
    ├── train2017       # .txt with TrainSet
    │   ├── 000050.txt
    │   ├── 000051.txt
    │   └── 000052.txt
    └── val2017         # .txt with ValSet
        ├── 001800.txt
        ├── 001801.txt
        └── 001802.txt
```
  
</details>  

## Android_demo 

This is a Redmi phone, the processor is Snapdragon 730G, and yolov5-lite is used for detection. The performance is as follows:

link: https://github.com/ppogg/YOLOv5-Lite/tree/master/ncnn_Android

Android_v5Lite-s: https://drive.google.com/file/d/1CtohY68N2B9XYuqFLiTp-Nd2kFWgAUR/view?us=sharing

Android_v5Lite-g: https://drive.google.com/file/d/1FnvkWxxP_aZwhi000xjIuhJ_OhOUJcj/view?us=sharing

## More detailed explanation

Detailed model link:
 [1] https://zhuanlan.zhihu.com/p/400545131
 
 [2] https://zhuanlan.zhihu.com/p/400545131
 
 [3] https://blog.csdn.net/weixin_45829462/article/details/119787840

## Reference

https://github.com/ultralytics/yolov5

https://github.com/megvii-model/ShuffleNet-Series

https://github.com/Tencent/ncnn
