
# YOLOv5 Lite OpenVINO Inference

**The following components are required.**
-	OpenVINO 2021.3
-	Python >=  3.6.0
-	Cmake >= 2.20.1
-	Opencv >= 4.2.0

###	Download Pytorch Weights 
v5lite-c.pt:  [https://drive.google.com/file/d/1H-GrrBNNO2_dpBjhd2R/view?usp=sharing](https://drive.google.com/file/d/1H-GrrxKVY9pH1-BNNO2_dhd2R/view?p=sharing)

###	Convert to ONNX Weights 

```bash
$ python models/export.py --weights v5lite-c.pt --img 640 --batch 1
$ python -m onnxsim v5lite-c.onnx v5lite-c-sim.onnx
```
###	Convert to IR model
```bash
python mo.py --input_model v5lite-c.onnx -s 255 --data_type FP16  --reverse_input_channels --output Conv_462,Conv_478,Conv_494
```
###	model_zoo:
v5lite-c.xml：[https://drive.google.com/file/d/1hlDPWmhvte6deiNgMRhuuvgnyKkYeC68/view?usp=sharing](https://drive.google.com/file/d/1hlDPWmhvte6deiNgMRhuuvgnyKkYeC68/view?usp=sharing)
v5lite-c.bin：[https://drive.google.com/file/d/1SStWwHQ8mQZ0UVfSTB3i9W580-5O6jX3/view?usp=sharing](https://drive.google.com/file/d/1SStWwHQ8mQZ0UVfSTB3i9W580-5O6jX3/view?usp=sharing)

###	Inference

```bash
python openvino.py -m v5lite-c.xml -i bike.jpg
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/21fa3eda9f89431ca3aef66fb15941cd.jpg?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAcG9nZ18=,size_20,color_FFFFFF,t_70,g_se,x_16)


