## model_zoo

### mnn-fp16(include mnnd&mnne)

google share: [https://drive.google.com/drive/folders/1PpFoZ4b8mVs1GmMxgf0WUtXUWaGK_JZe?usp=sharing](https://drive.google.com/drive/folders/1PpFoZ4b8mVs1GmMxgf0WUtXUWaGK_JZe?usp=sharing)

baidu share: [https://pan.baidu.com/s/1Um163Rf-9ezqHBvZHwzyWg?pwd=pogg](https://pan.baidu.com/s/1Um163Rf-9ezqHBvZHwzyWg?pwd=pogg)

### mnn-int8(include mnnd&mnne)

google share: [https://drive.google.com/drive/folders/1mSU8g94c77KKsHC-07p5V3tJOZYPQ-g6?usp=sharing](https://drive.google.com/drive/folders/1mSU8g94c77KKsHC-07p5V3tJOZYPQ-g6?usp=sharing)

baidu share: [https://pan.baidu.com/s/1V7LVt7AxyG7HjlCJJoOJ3A?pwd=pogg](https://pan.baidu.com/s/1V7LVt7AxyG7HjlCJJoOJ3A?pwd=pogg)

## Detection effect

Pytorch{320×320}：

<img src="https://github.com/ppogg/YOLOv5-Lite/assets/82716366/3ad8cb35-0a2e-4edf-af6c-ff0cf946f355" width="640" height="480"/><br/>

onnx{check fp32}@{320×320}:

<img src="https://github.com/ppogg/YOLOv5-Lite/assets/82716366/66a0f7e2-aaa2-4597-a419-10a193c19015" width="640" height="640"/><br/>

MNN{fp32}@{320×320}:

<img src="https://github.com/ppogg/YOLOv5-Lite/assets/82716366/7ab98964-05d4-42d0-a011-c464b457955d" width="640" height="480"/><br/>

### 15FPS can be used with yolov5 on the Raspberry Pi with only 0.1T computing power

#### Excluding the three minute warm-ups, the device temperature is stable above 65°, the forward reasoning framework is mnn

Show picture infer result (include imread - preprocess - network forward - postprocess - imwrite):

![8936b37d43e7ae283aa37b045227d8b](https://github.com/ppogg/YOLOv5-Lite/assets/82716366/29b4756d-8280-4caf-9aef-fe4dff8af79d)

Show camera infer result (include frame capture - imread - preprocess - network forward - postprocess - imshow):

![](https://pic2.zhimg.com/80/v2-55a619359357c1dda6d945866ac27855_720w.webp)

Memory of board when run the sdk:

![openmp](https://github.com/ppogg/YOLOv5-Lite/assets/82716366/6b904747-ccbd-4bc7-a451-ee551a426d77)

How to use:
```bash
$ mkdir build
$ cd build
$ cmake ..
$ make -j
$ main
```

