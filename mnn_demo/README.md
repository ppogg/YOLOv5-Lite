## 使用MNN部署YoloV5模型  

step1：  
```
install opecv
install protobuf
install cmake
```

step2：  
```
git clone https://github.com/alibaba/MNN.git
cd MNN
mkdir bulid && cd build
sudo cmake ..
sudo make
```

step3：  
```
cd mnn_demo
mkdir bulid && cd build
sudo cmake ..
sudo make
```

step4：  
```
./yolov5
```
![output](https://user-images.githubusercontent.com/82716366/135485823-d22486ac-ee5a-41a6-bec5-74116f0bcb47.jpg)

