#include <MNN/MNNDefine.h>
#include <MNN/MNNForwardType.h>
#include <MNN/Interpreter.hpp>
#include <MNN/ImageProcess.hpp>
#include <opencv2/opencv.hpp>
//#include <cv/cv.hpp>


typedef struct
{
    float x1, y1, x2, y2, score;
    int label;
} BoxInfo;

typedef struct
{
    int inpSize, maxSide, Padw, Padh;
    float ratio;
} MatInfo;

cv::Mat preprocess(cv::Mat &cv_mat, MatInfo &mmat_objection);

void nms(std::vector<BoxInfo> &result, float nms_threshold);

void draw_box(cv::Mat &cv_mat, std::vector<BoxInfo> &boxes, MatInfo &mmat_objection);
