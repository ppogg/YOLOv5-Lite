The yolov5-lite object detection

This is a sample ncnn android project, it depends on ncnn library and opencv

https://github.com/Tencent/ncnn

https://github.com/nihui/opencv-mobile

## model_zoo

yolov5-lite.bin: [https://drive.google.com/file/d/1pR_hFmhmI19Pev_t1i75z7btfJTY3m2f/view?usp=sharing](https://drive.google.com/file/d/1pR_hFmhmI19Pev_t1i75z7btfJTY3m2f/view?usp=sharing)

yolov5-lite-int8.bin: [https://drive.google.com/file/d/1U4Vt6Oqa7ER0CahaMIdHcn0eagv1iOcw/view?usp=sharing](https://drive.google.com/file/d/1U4Vt6Oqa7ER0CahaMIdHcn0eagv1iOcw/view?usp=sharing)


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
wget yolov5-lite-i8.bin or yolov5-lite.bin into assets
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
<img src="https://user-images.githubusercontent.com/82716366/130357030-c4131b64-55e4-40c9-9f66-c17b42d2409b.jpg" width="300"/><br/>

## reference  
https://github.com/nihui/ncnn-android-yolov5
https://github.com/FeiGeChuanShu/ncnn-android-yolox  
https://github.com/ppogg/YOLOv5-Lite
