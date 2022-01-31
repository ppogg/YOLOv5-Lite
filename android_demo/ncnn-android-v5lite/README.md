The yolov5-lite object detection

This is a sample ncnn android project, it depends on ncnn library and opencv

https://github.com/Tencent/ncnn

https://github.com/nihui/opencv-mobile

## model_zoo

https://github.com/ppogg/ncnn-android-v5lite/tree/master/app/src/main/assets


## how to build and run
### step1
https://github.com/Tencent/ncnn/releases

* Download ncnn-YYYYMMDD-android-vulkan.zip or build ncnn for android yourself
* Extract ncnn-YYYYMMDD-android-vulkan.zip into **app/src/main/jni** and change the **ncnn_DIR** path to yours in **app/src/main/jni/CMakeLists.txt**

### step2
https://github.com/nihui/opencv-mobile

* Download opencv-mobile-XYZ-android.zip
* Extract opencv-mobile-XYZ-android.zip into **app/src/main/jni** and change the **OpenCV_DIR** path to yours in **app/src/main/jni/CMakeLists.txt**

### step3
```
cd ncnn_Android/ncnn-android-yolov5/app/src/main/assets
wget all the *.param and *.bin
```

### step4
* Open this project with Android Studio, build it and enjoy!

## some notes
* Android ndk camera is used for best efficiency
* Crash may happen on very old devices for lacking HAL3 camera interface
* All models are manually modified to accept dynamic input shape
* Most small models run slower on GPU than on CPU, this is common
* FPS may be lower in dark environment because of longer camera exposure time

## screenshot
<img src="https://user-images.githubusercontent.com/82716366/151705519-de3ad1f1-e297-4125-989a-04e49dcf2876.jpg" width="600"/><br/>

<img src="https://pic1.zhimg.com/80/v2-c013df3638fd41d10103ea259b18e588_720w.jpg" width="300"/><br/>

## reference  
https://github.com/nihui/ncnn-android-yolov5

https://github.com/FeiGeChuanShu/ncnn-android-yolox  

https://github.com/ppogg/YOLOv5-Lite
