
#include <iostream>
#include <string>
#include <ctime>
#include <stdio.h>
#include <omp.h>

#include <MNN/MNNDefine.h>
#include <MNN/MNNForwardType.h>
#include <MNN/Interpreter.hpp>

#include "utils.h"
#define use_camera 0
#define mnnd 1

std::vector<BoxInfo> decode(cv::Mat &cv_mat, std::shared_ptr<MNN::Interpreter> &net, MNN::Session *session, int INPUT_SIZE)
{
    std::vector<int> dims{1, INPUT_SIZE, INPUT_SIZE, 3};
    auto nhwc_Tensor = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
    auto nhwc_data = nhwc_Tensor->host<float>();
    auto nhwc_size = nhwc_Tensor->size();
    std::memcpy(nhwc_data, cv_mat.data, nhwc_size);

    auto inputTensor = net->getSessionInput(session, nullptr);
    inputTensor->copyFromHostTensor(nhwc_Tensor);

    net->runSession(session);
    MNN::Tensor *tensor_scores = net->getSessionOutput(session, "outputs");
    MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());
    tensor_scores->copyToHostTensor(&tensor_scores_host);
    auto pred_dims = tensor_scores_host.shape();

#if mnnd
    const unsigned int num_proposals = pred_dims.at(1);
    const unsigned int num_classes = pred_dims.at(2) - 5;
    std::vector<BoxInfo> bbox_collection;

    for (unsigned int i = 0; i < num_proposals; ++i)
    {
        const float *offset_obj_cls_ptr = tensor_scores_host.host<float>() + (i * (num_classes + 5)); // row ptr
        float obj_conf = offset_obj_cls_ptr[4];
        if (obj_conf < 0.5)
            continue;

        float cls_conf = offset_obj_cls_ptr[5];
        unsigned int label = 0;
        for (unsigned int j = 0; j < num_classes; ++j)
        {
            float tmp_conf = offset_obj_cls_ptr[j + 5];
            if (tmp_conf > cls_conf)
            {
                cls_conf = tmp_conf;
                label = j;
            }
        }

        float conf = obj_conf * cls_conf; 
        if (conf < 0.50)
            continue;

        float cx = offset_obj_cls_ptr[0];
        float cy = offset_obj_cls_ptr[1];
        float w = offset_obj_cls_ptr[2];
        float h = offset_obj_cls_ptr[3];

        float x1 = (cx - w / 2.f);
        float y1 = (cy - h / 2.f);
        float x2 = (cx + w / 2.f);
        float y2 = (cy + h / 2.f);

        BoxInfo box;
        box.x1 = std::max(0.f, x1);
        box.y1 = std::max(0.f, y1);
        box.x2 = std::min(x2, (float)INPUT_SIZE - 1.f);
        box.y2 = std::min(y2, (float)INPUT_SIZE - 1.f);
        box.score = conf;
        box.label = label;
        bbox_collection.push_back(box);
    }
#else
    const unsigned int num_proposals = pred_dims.at(0);
    const unsigned int num_datainfo = pred_dims.at(1);
    std::vector<BoxInfo> bbox_collection;
    for (unsigned int i = 0; i < num_proposals; ++i)
    {
        const float *offset_obj_cls_ptr = tensor_scores_host.host<float>() + (i * num_datainfo); // row ptr
        float obj_conf = offset_obj_cls_ptr[4];
        if (obj_conf < 0.5)
            continue;

        float x1 = offset_obj_cls_ptr[0];
        float y1 = offset_obj_cls_ptr[1];
        float x2 = offset_obj_cls_ptr[2];
        float y2 = offset_obj_cls_ptr[3];

        BoxInfo box;
        box.x1 = std::max(0.f, x1);
        box.y1 = std::max(0.f, y1);
        box.x2 = std::min(x2, (float)INPUT_SIZE - 1.f);
        box.y2 = std::min(y2, (float)INPUT_SIZE - 1.f);
        box.score = offset_obj_cls_ptr[4];
        box.label = offset_obj_cls_ptr[5];
        bbox_collection.push_back(box);
    }
#endif
    delete nhwc_Tensor;
    return bbox_collection;
}

int main(int argc, char const *argv[])
{

    std::string model_name = "../models/v5lite-e-mnnd_fp16.mnn";

    std::shared_ptr<MNN::Interpreter> net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_name.c_str()));
    if (nullptr == net)
    {
        return 0;
    }

    MNN::ScheduleConfig config;
    config.numThread = 4;
    config.type = static_cast<MNNForwardType>(MNN_FORWARD_CPU);
    MNN::BackendConfig backendConfig;
    // backendConfig.precision = (MNN::BackendConfig::PrecisionMode)1;
    backendConfig.precision = MNN::BackendConfig::Precision_Low_BF16;
    config.backendConfig = &backendConfig;
    MNN::Session *session = net->createSession(config);

    std::vector<BoxInfo> bbox_collection;
    cv::Mat image;
    MatInfo mmat_objection;
    mmat_objection.inpSize = 320;

#if use_camera
    cv::VideoCapture capture;
    capture.open(0);

    cv::Mat frame;
    while (true)
    {
        bbox_collection.clear();
        
        struct timespec begin, end;
        long time;
        clock_gettime(CLOCK_MONOTONIC, &begin);
        
        capture >> frame;
        cv::Mat raw_image = frame;

        cv::Mat pimg = preprocess(raw_image, mmat_objection);
        bbox_collection = decode(pimg, net, session, mmat_objection.inpSize);
        nms(bbox_collection, 0.50);
        draw_box(raw_image, bbox_collection, mmat_objection);

        clock_gettime(CLOCK_MONOTONIC, &end);
        time = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec);
        if(time > 0) printf(">> Time : %lf ms\n", (double)time / 1000000);
    }
#else
    for (size_t i = 0; i < 100; i++)
    {
        bbox_collection.clear();

        struct timespec begin, end;
        long time;
        clock_gettime(CLOCK_MONOTONIC, &begin);

        std::string image_name = "../images/000000001000.jpg";
        cv::Mat raw_image = cv::imread(image_name.c_str());

        cv::Mat pimg = preprocess(raw_image, mmat_objection);

        bbox_collection = decode(pimg, net, session, mmat_objection.inpSize);

        nms(bbox_collection, 0.50);

        draw_box(raw_image, bbox_collection, mmat_objection);

        clock_gettime(CLOCK_MONOTONIC, &end);
        time = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec);
        if(time > 0) printf(">> Time : %lf ms\n", (double)time / 1000000);
    }
#endif
    return 0;
}
