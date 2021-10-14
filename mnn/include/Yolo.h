
#ifndef Yolo_H
#define Yolo_H

#include <MNN/MNNDefine.h>
#include <MNN/MNNForwardType.h>
#include <MNN/Interpreter.hpp>

#include "util.h"

namespace yolocv {
    typedef struct {
        int width;
        int height;
    } YoloSize;
}

typedef struct {
    std::string name;
    int stride;
    std::vector<yolocv::YoloSize> anchors;
} YoloLayerData;

class BoxInfo
{
public:
    int x1,y1,x2,y2,label,id;
    float score;
};

 std::vector<BoxInfo>
    decode_infer(MNN::Tensor & data, int stride, const yolocv::YoloSize &frame_size, int net_size, int num_classes,
                 const std::vector<yolocv::YoloSize> &anchors, float threshold);

     void nms(std::vector<BoxInfo> &result, float nms_threshold);



#endif //Yolo_H
