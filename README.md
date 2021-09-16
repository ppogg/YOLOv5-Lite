### Update on 2021-09-16 [YOLOv5 add Repvgg Block]

![repyolov5](https://user-images.githubusercontent.com/82716366/133568390-918822c8-ff7c-43bd-bd54-c999d3e57f01.png)

# YOLOv5-Lite：lighter, faster and easier to deploy   ![](https://zenodo.org/badge/DOI/10.5281/zenodo.5241425.svg)

![0111](https://user-images.githubusercontent.com/82716366/129756605-a0cba66c-b296-43f1-b83e-39f5f10cd1c2.jpg)


Perform a series of ablation experiments on yolov5 to make it lighter (smaller Flops, lower memory, and fewer parameters) and faster (add shuffle channel, yolov5 head for channel reduce. It can infer at least 10+ FPS On the Raspberry Pi 4B when input the frame with 320×320) and is easier to deploy (removing the Focus layer and four slice operations, reducing the model quantization accuracy to an acceptable range).

### Comparison of ablation experiment results

  ID|Model | Input_size|Flops| Params | Size（M） |Map@0.5|Map@.5:0.95
 :-----:|:-----:|:-----:|:----------:|:----:|:----:|:----:|:----:|
001| yolo-fastest| 320×320|0.25G|0.35M|1.4| 24.4| -
002| nanodet-m| 320×320| 0.72G|0.95M|1.8|- |20.6
003| yolo-fastest-xl| 320×320|0.72G|0.92M|3.5| 34.3| -
004| yolov5-lite| 320×320|1.43G |1.62M|3.3| 36.2|20.8| 
005| yolov3-tiny| 416×416| 6.96G|6.06M|23.0| 33.1|16.6
006| yolov4-tiny| 416×416| 5.62G|8.86M| 33.7|40.2|21.7
007| nanodet-m| 416×416| 1.2G	|0.95M|1.8|- |23.5
008| yolov5-lite| 416×416|2.42G |1.62M|3.3| 41.3|24.4| 
009| yolov5-lite| 640×640|4.12G |1.62M|3.3| 45.7|27.1| 
010| yolov5s| 640×640| 17.0G|7.3M|14.2| 55.9|36.2

### Comparison on different platforms

Equipment|Computing backend|System|Framework|Input|Speed{our}|Speed{yolov5s}
:---:|:---:|:---:|:---:|:---:|:---:|:---:
Inter|@i5-10210U|window(x86)|640×640|torch-cpu|112ms|179ms
Nvidia|@RTX 2080Ti|Linux(x86)|640×640|torch-gpu|11ms|13ms
Raspberrypi 4B|@ARM Cortex-A72|Linux(arm64)|320×320|ncnn|97ms|371ms

## Detection effect 

<img src="https://user-images.githubusercontent.com/82716366/133584299-32c19883-2eb2-48ef-a22c-34e244d0ffbe.jpg" width="900" /><br/>

## Base on YOLOv5 -------------------

<img src="https://user-images.githubusercontent.com/82716366/133585711-22368708-09d6-4a1e-bda8-546139392434.jpg" width="900" /><br/>

## Android_demo ---------------------

This is a Redmi phone, the processor is Snapdragon 730G, and yolov5-lite is used for detection. The performance is as follows:

<img src="https://user-images.githubusercontent.com/82716366/130217501-6db77073-7727-4ed8-89fe-e644c4bf8cf9.jpg" width="600" /><br/>

Outdoor scene example:

<img src="https://user-images.githubusercontent.com/82716366/130357030-c4131b64-55e4-40c9-9f66-c17b42d2409b.jpg" width="400"/><br/>


### More detailed explanation ----------------------

Detailed model link: https://zhuanlan.zhihu.com/p/400545131

![image](https://user-images.githubusercontent.com/82716366/129891972-31f230e3-6e30-4392-820e-6aef08a51ab1.png)



## Reference

https://github.com/Tencent/ncnn

https://github.com/ultralytics/yolov5

https://github.com/megvii-model/ShuffleNet-Series
