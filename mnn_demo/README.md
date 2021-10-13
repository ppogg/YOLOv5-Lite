## 使用MNN部署YoloV5模型  

#### step1：  
```
install opecv
install protobuf
install cmake
```

#### step2：  
```
git clone https://github.com/alibaba/MNN.git
cd MNN
mkdir bulid && cd build
sudo cmake ..
sudo make
```

#### step3：  
```
cd mnn_demo
mkdir bulid && cd build
sudo cmake ..
sudo make
```

#### step4：  
```
mkdir model_zoo && cd model_zoo
wget v5lite-s.mnn or v5lite-s-int4.mnn into model_zoo
```

v5lite-s.mnn: [https://drive.google.com/file/d/10dBsY0T19Kyz2sZ4ebfpsb6dnG58pmYq/view?usp=sharing](https://drive.google.com/file/d/10dBsY0T19Kyz2sZ4ebfpsb6dnG58pmYq/view?usp=sharing)

v5lite-s-int4.mnn: [https://drive.google.com/file/d/1v90z5sWx6rTnrF9jejugZup2YuIuXObR/view?usp=sharing](https://drive.google.com/file/d/1v90z5sWx6rTnrF9jejugZup2YuIuXObR/view?usp=sharing)

#### step5：  
```
./yolov5
```
![output](https://user-images.githubusercontent.com/82716366/135485823-d22486ac-ee5a-41a6-bec5-74116f0bcb47.jpg)

## Reference

https://github.com/techshoww/mnn-yolov5
