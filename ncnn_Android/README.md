The YOLOv5（shufflev2-yolov5） object detection

This is a sample ncnn android project, it depends on ncnn library and opencv

https://github.com/Tencent/ncnn

https://github.com/nihui/opencv-mobile


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
* Open this project with Android Studio, build it and enjoy!

## some notes
* Android ndk camera is used for best efficiency
* Crash may happen on very old devices for lacking HAL3 camera interface
* All models are manually modified to accept dynamic input shape
* Most small models run slower on GPU than on CPU, this is common
* FPS may be lower in dark environment because of longer camera exposure time
* Phone heating or insufficient battery will cause the detection speed to decrease
* Prone to misdetection under low light conditions
* 
## screenshot

This is a Redmi phone, the processor is Snapdragon 730G, and shufflev2-yolov5 is used for detection. The performance is as follows:

<img src="https://user-images.githubusercontent.com/82716366/130217501-6db77073-7727-4ed8-89fe-e644c4bf8cf9.jpg" width="700" height="350"/><br/>

This is the quantized int8 model:

<img src="https://user-images.githubusercontent.com/82716366/130217583-d645ae5b-4f48-49dc-8672-dd60a055a67e.jpg" width="700" height="350"/><br/>

Outdoor scene example:

<img src="https://user-images.githubusercontent.com/82716366/130357030-c4131b64-55e4-40c9-9f66-c17b42d2409b.jpg" width="400"/><br/>

## reference  
https://github.com/nihui/ncnn-android-yolov5

https://github.com/FeiGeChuanShu/ncnn-android-yolox  

https://github.com/ppogg/shufflev2-yolov5 

