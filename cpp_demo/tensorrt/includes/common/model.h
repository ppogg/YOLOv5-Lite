//
// Created by linghu8812 on 2021/2/8.
//

#ifndef TENSORRT_INFERENCE_MODEL_H
#define TENSORRT_INFERENCE_MODEL_H

#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "common.h"

class Model
{
public:
    void LoadEngine();
    virtual bool InferenceFolder(const std::string &folder_name) = 0;

protected:
    bool readTrtFile();
    virtual void EngineInference(const std::vector<std::string> &image_list, const int &outSize,void **buffers,
            const std::vector<int64_t> &bufferSize, cudaStream_t stream) = 0;
    virtual std::vector<float> prepareImage(std::vector<cv::Mat> & image) = 0;
    std::string engine_file;
    std::string labels_file;
    int BATCH_SIZE;
    int INPUT_CHANNEL;
    int IMAGE_WIDTH;
    int IMAGE_HEIGHT;
    nvinfer1::ICudaEngine *engine;
    nvinfer1::IExecutionContext *context;
    std::vector<float> img_mean;
    std::vector<float> img_std;
};

#endif //TENSORRT_INFERENCE_MODEL_H
