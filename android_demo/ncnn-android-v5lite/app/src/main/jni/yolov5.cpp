// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "yolov5.h"
#include <benchmark.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "omp.h"
#include "cpu.h"


static inline float intersection_area(const Object &a, const Object &b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j) {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object> &faceobjects) {
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked,
                              float nms_threshold) {
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++) {
        const Object &a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int) picked.size(); j++) {
            const Object &b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

// sigmoid
static inline float sigmoid(float x) {
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

// unsigmoid
static inline float unsigmoid(float y) {
    return static_cast<float>(-1.0 * (log((1.0 / y) - 1.0)));
}

static void generate_proposals(const ncnn::Mat &anchors, int stride, const ncnn::Mat &in_pad,
                               const ncnn::Mat &feat_blob, float prob_threshold,
                               std::vector <Object> &objects) {
    const int num_grid = feat_blob.h;
    if (prob_threshold > 0.6)
        float unsig_pro = unsigmoid(prob_threshold);

    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h) {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    } else {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = feat_blob.w - 5;

    const int num_anchors = anchors.w / 2;

    for (int q = 0; q < num_anchors; q++) {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        const ncnn::Mat feat = feat_blob.channel(q);

        for (int i = 0; i < num_grid_y; i++) {
            for (int j = 0; j < num_grid_x; j++) {
                const float *featptr = feat.row(i * num_grid_x + j);

                // find class index with max class score
                int class_index = 0;
                float class_score = -FLT_MAX;
                float box_score = featptr[4];
                if (prob_threshold > 0.6) {
                    // while prob_threshold > 0.6, unsigmoid better than sigmoid
                    if (box_score > unsig_pro) {
                        for (int k = 0; k < num_class; k++) {
                            float score = featptr[5 + k];
                            if (score > class_score) {
                                class_index = k;
                                class_score = score;
                            }
                        }

                        float confidence = sigmoid(box_score) * sigmoid(class_score);

                        if (confidence >= prob_threshold) {

                            float dx = sigmoid(featptr[0]);
                            float dy = sigmoid(featptr[1]);
                            float dw = sigmoid(featptr[2]);
                            float dh = sigmoid(featptr[3]);

                            float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                            float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                            float pb_w = pow(dw * 2.f, 2) * anchor_w;
                            float pb_h = pow(dh * 2.f, 2) * anchor_h;

                            float x0 = pb_cx - pb_w * 0.5f;
                            float y0 = pb_cy - pb_h * 0.5f;
                            float x1 = pb_cx + pb_w * 0.5f;
                            float y1 = pb_cy + pb_h * 0.5f;

                            Object obj;
                            obj.rect.x = x0;
                            obj.rect.y = y0;
                            obj.rect.width = x1 - x0;
                            obj.rect.height = y1 - y0;
                            obj.label = class_index;
                            obj.prob = confidence;

                            objects.push_back(obj);
                        }
                    } else {
                        for (int k = 0; k < num_class; k++) {
                            float score = featptr[5 + k];
                            if (score > class_score) {
                                class_index = k;
                                class_score = score;
                            }
                        }
                        float confidence = sigmoid(box_score) * sigmoid(class_score);

                        if (confidence >= prob_threshold) {
                            float dx = sigmoid(featptr[0]);
                            float dy = sigmoid(featptr[1]);
                            float dw = sigmoid(featptr[2]);
                            float dh = sigmoid(featptr[3]);

                            float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                            float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                            float pb_w = pow(dw * 2.f, 2) * anchor_w;
                            float pb_h = pow(dh * 2.f, 2) * anchor_h;

                            float x0 = pb_cx - pb_w * 0.5f;
                            float y0 = pb_cy - pb_h * 0.5f;
                            float x1 = pb_cx + pb_w * 0.5f;
                            float y1 = pb_cy + pb_h * 0.5f;

                            Object obj;
                            obj.rect.x = x0;
                            obj.rect.y = y0;
                            obj.rect.width = x1 - x0;
                            obj.rect.height = y1 - y0;
                            obj.label = class_index;
                            obj.prob = confidence;

                            objects.push_back(obj);
                        }
                    }
                }
            }
        }
    }
}


Yolov5::Yolov5() {
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int Yolov5::load(const char *modeltype, int _target_size, const float *_mean_vals,
                 const float *_norm_vals, bool use_gpu) {
    yolov5.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolov5.opt = ncnn::Option();

#if NCNN_VULKAN
    yolov5.opt.use_vulkan_compute = use_gpu;
#endif
    yolov5.opt.num_threads = ncnn::get_big_cpu_count();
    yolov5.opt.blob_allocator = &blob_pool_allocator;
    yolov5.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    yolov5.load_param(parampath);
    yolov5.load_model(modelpath);

    target_size = _target_size;

    mean_vals[0] = 1 / 255.f;
    mean_vals[1] = 1 / 255.f;
    mean_vals[2] = 1 / 255.f;
    norm_vals[0] = 0;
    norm_vals[1] = 0;
    norm_vals[2] = 0;

    return 0;
}

int Yolov5::load(AAssetManager *mgr, const char *modeltype, int _target_size, bool use_gpu) {
    yolov5.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolov5.opt = ncnn::Option();
#if NCNN_VULKAN
    yolov5.opt.use_vulkan_compute = use_gpu;
#endif
    yolov5.opt.num_threads = ncnn::get_big_cpu_count();
    yolov5.opt.blob_allocator = &blob_pool_allocator;
    yolov5.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];

    if (modeltype == "320-lite-e") {
        target_size = 320;
        sprintf(parampath, "%s.param", "e");
        sprintf(modelpath, "%s.bin", "e");
    } else if (modeltype == "416-lite-e") {
        target_size = 416;
        sprintf(parampath, "%s.param", "e");
        sprintf(modelpath, "%s.bin", "e");
    } else if (modeltype == "320-lite-i8e") {
        target_size = 320;
        sprintf(parampath, "%s.param", "i8e");
        sprintf(modelpath, "%s.bin", "i8e");
    } else if (modeltype == "416-lite-i8e") {
        target_size = 416;
        sprintf(parampath, "%s.param", "i8e");
        sprintf(modelpath, "%s.bin", "i8e");
    }else if (modeltype == "416-lite-s") {
        target_size = 416;
        sprintf(parampath, "%s.param", "s");
        sprintf(modelpath, "%s.bin", "s");
    } else if (modeltype == "416-lite-i8s") {
        target_size = 416;
        sprintf(parampath, "%s.param", "i8s");
        sprintf(modelpath, "%s.bin", "i8s");
    } else if (modeltype == "512-lite-c") {
        target_size = 512;
        sprintf(parampath, "%s.param", "c");
        sprintf(modelpath, "%s.bin", "c");
    }
//    target_size = 320;

    yolov5.load_param(mgr, parampath);
    yolov5.load_model(mgr, modelpath);

    return 0;
}


int Yolov5::detect(const cv::Mat &rgb, std::vector<Object> &objects, float prob_threshold,
                   float nms_threshold) {
    int img_w = rgb.cols;
    int img_h = rgb.rows;

    // letterbox pad to multiple of 32
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h) {
        scale = (float) target_size / w;
        w = target_size;
        h = h * scale;
    } else {
        scale = (float) target_size / h;
        h = target_size;
        w = w * scale;
    }


    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, img_w, img_h,
                                                 w, h);

    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    double t1, t2, t3, t4, t5, t6, t;
    int wpad = target_size - w;//(w + 31) / 32 * 32 - w;
    int hpad = target_size - h;//(h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, 0, hpad, 0, wpad, ncnn::BORDER_CONSTANT, 114.f);

    // so for 0-255 input image, rgb_mean should multiply 255 and norm should div by std.

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolov5.create_extractor();

    ex.input("images", in_pad);

    std::vector<Object> proposals;

    // stride 8
    {
        ncnn::Mat out;
        t1 = ncnn::get_current_time();
        ex.extract("output", out);
        t2 = ncnn::get_current_time() - t1;

        ncnn::Mat anchors(6);
        anchors[0] = 10.f;
        anchors[1] = 13.f;
        anchors[2] = 16.f;
        anchors[3] = 30.f;
        anchors[4] = 33.f;
        anchors[5] = 23.f;

        std::vector<Object> objects8;
        generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat out;
        t3 = ncnn::get_current_time();
        ex.extract("1111", out);
        t4 = ncnn::get_current_time() - t3;

        ncnn::Mat anchors(6);
        anchors[0] = 30.f;
        anchors[1] = 61.f;
        anchors[2] = 62.f;
        anchors[3] = 45.f;
        anchors[4] = 59.f;
        anchors[5] = 119.f;

        std::vector<Object> objects16;
        generate_proposals(anchors, 16, in_pad, out, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat out;
        t5 = ncnn::get_current_time();
        ex.extract("2222", out);
        t6 = ncnn::get_current_time() - t5;

        ncnn::Mat anchors(6);
        anchors[0] = 116.f;
        anchors[1] = 90.f;
        anchors[2] = 156.f;
        anchors[3] = 198.f;
        anchors[4] = 373.f;
        anchors[5] = 326.f;

        std::vector<Object> objects32;
        generate_proposals(anchors, 32, in_pad, out, prob_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }

    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    objects.resize(count);
    #pragma omp parallel for num_threads(ncnn::get_big_cpu_count())
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x) / scale;
        float y0 = (objects[i].rect.y) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

        // clip
        x0 = std::max(std::min(x0, (float) (img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float) (img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float) (img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float) (img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    struct {
        bool operator()(const Object &a, const Object &b) const {
            return a.rect.area() > b.rect.area();
        }
    } objects_area_greater;
    std::sort(objects.begin(), objects.end(), objects_area_greater);

    t = (t2 + t4 + t6);
    return t;
}

//int Yolov5::drawradrect(cv::Mat &rgb, cv::Point topLeft, cv::Size rectSz, const cv::Scalar lineColor, const int thickness, const int lineType, const float cornerCurvatureRatio) {
//    // corners:
//    // p1 - p2
//    // |     |
//    // p4 - p3
//    //
//    cv::Point p1 = topLeft;
//    cv::Point p2 = cv::Point(p1.x + rectSz.width, p1.y);
//    cv::Point p3 = cv::Point(p1.x + rectSz.width, p1.y + rectSz.height);
//    cv::Point p4 = cv::Point(p1.x, p1.y + rectSz.height);
//    float cornerRadius = rectSz.height * cornerCurvatureRatio;
//
//
//    // draw straight lines
//    cv::line(rgb, cv::Point(p1.x + cornerRadius, p1.y), cv::Point(p2.x - cornerRadius, p2.y), lineColor, thickness, lineType);
//    cv::line(rgb, cv::Point(p2.x, p2.y + cornerRadius), cv::Point(p3.x, p3.y - cornerRadius), lineColor, thickness, lineType);
//    cv::line(rgb, cv::Point(p4.x + cornerRadius, p4.y), cv::Point(p3.x - cornerRadius, p3.y), lineColor, thickness, lineType);
//    cv::line(rgb, cv::Point(p1.x, p1.y + cornerRadius), cv::Point(p4.x, p4.y - cornerRadius), lineColor, thickness, lineType);
//
//    // draw arcs
//    cv::Size rad = cv::Size(cornerRadius, cornerRadius);
//    cv::ellipse(rgb, p1 + cv::Point(cornerRadius, cornerRadius), rad, 180.0, 0, 90, lineColor, thickness, lineType);
//    cv::ellipse(rgb, p2 + cv::Point(-cornerRadius, cornerRadius), rad, 270.0, 0, 90, lineColor, thickness, lineType);
//    cv::ellipse(rgb, p3 + cv::Point(-cornerRadius, -cornerRadius), rad, 0.0, 0, 90, lineColor, thickness, lineType);
//    cv::ellipse(rgb, p4 + cv::Point(cornerRadius, -cornerRadius), rad, 90.0, 0, 90, lineColor, thickness, lineType);
//
//    return 0;
//}

int Yolov5::draw(cv::Mat &rgb, const std::vector<Object> &objects) {
    static const char *class_names[] = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat",
            "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
            "horse",
            "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
            "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
            "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
            "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear",
            "hair drier", "toothbrush"
    };
    static const unsigned char colors[19][3] = {
            {54,  67,  244},
            {99,  30,  233},
            {176, 39,  156},
            {183, 58,  103},
            {181, 81,  63},
            {243, 150, 33},
            {244, 169, 3},
            {212, 188, 0},
            {136, 150, 0},
            {80,  175, 76},
            {74,  195, 139},
            {57,  220, 205},
            {59,  235, 255},
            {7,   193, 255},
            {0,   152, 255},
            {34,  87,  255},
            {72,  85,  121},
            {158, 158, 158},
            {139, 125, 96}
    };

    int color_index = 0;
    #pragma omp parallel for num_threads(ncnn::get_big_cpu_count())
    for (size_t i = 0; i < objects.size(); i++) {
        const Object &obj = objects[i];

        const unsigned char *color = colors[color_index % 19];
        color_index++;

        cv::Scalar cc(color[0], color[1], color[2]);

        cv::rectangle(rgb, obj.rect, cc, 2);

        char text[256];
        sprintf(text, "%s %.1f%% %.1f %.1f", class_names[obj.label], obj.prob * 100, obj.box_score * 100, obj.unsig_pro * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1,
                                              &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y),
                                    cv::Size(label_size.width, label_size.height + baseLine)),
                      cc, -1);
//        drawradrect(rgb, cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine), cc, 2, 1, 2.5);
        cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381) ? cv::Scalar(0, 0, 0)
                                                                    : cv::Scalar(255, 255, 255);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    textcc, 1);

    }


    return 0;
}
