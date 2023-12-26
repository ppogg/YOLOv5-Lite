#include <fstream>
#include <sys/time.h>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
// #include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace cv;
using namespace Ort;

#define use_camera 0

static const char *class_names[] = {
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
	"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
	"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
	"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
	"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
	"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
	"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
	"hair drier", "toothbrush"};

struct Net_config
{
	float confThreshold; // Confidence threshold
	string modelpath;
};

class v5Lite
{
public:
	v5Lite(Net_config config);
	void detect(Mat &frame, Net_config config);

private:
	int inpWidth, inpHeight, maxSide, Padw, Padh;
	float ratio;

	int nout, num_proposal;

	vector<float> input_image_;
	Mat letter_(Mat &img);
	void normalize_(Mat img);
	bool has_postprocess;

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "v5Lite");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char *> input_names;
	vector<char *> output_names;
	vector<vector<int64_t>> input_node_dims;  // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

v5Lite::v5Lite(Net_config config)
{
	string model_path = config.modelpath;
	// std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	// OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	// ort_session = new Session(env, widestr.c_str(), sessionOptions);
	// window type
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	// OrtCUDAProviderOptions cuda_options{};
	// sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
	ort_session = new Session(env, model_path.c_str(), sessionOptions);
	// linux type

	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
}

void v5Lite::normalize_(Mat img)
{
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];
				this->input_image_[c * row * col + i * col + j] = pix / 255.0;
			}
		}
	}
}

Mat v5Lite::letter_(Mat &img)
{
	Mat dstimg;
	this->maxSide = img.rows > img.cols ? img.rows : img.cols;
	this->ratio = this->inpWidth / float(this->maxSide);
	int fx = int(img.cols * ratio);
	int fy = int(img.rows * ratio);
	this->Padw = int((this->inpWidth - fx) * 0.5);
	this->Padh = int((this->inpHeight - fy) * 0.5);
	resize(img, dstimg, Size(fx, fy));
	copyMakeBorder(dstimg, dstimg, Padh, Padh, Padw, Padw, BORDER_CONSTANT, Scalar::all(127));
	return dstimg;
}

void v5Lite::detect(Mat &frame, Net_config config)
{
	Mat dstimg = this->letter_(frame);
	this->normalize_(dstimg);
	array<int64_t, 4> input_shape_{1, 3, this->inpHeight, this->inpWidth};

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
	vector<Value> ort_outputs = ort_session->Run(RunOptions{nullptr}, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());

	Ort::Value &predictions = ort_outputs.at(0);
	auto pred_dims = predictions.GetTensorTypeAndShapeInfo().GetShape();
	num_proposal = pred_dims.at(0);
	nout = pred_dims.at(1);
	const float *pdata = predictions.GetTensorMutableData<float>();

	for (int n = 0; n < this->num_proposal; n++)
	{
		float box_score = pdata[4 + nout * n];
		if (box_score > config.confThreshold)
		{
			int xmin = int(pdata[n * nout] - this->Padw) * (1.0 / this->ratio);
			int ymin = int(pdata[n * nout + 1] - this->Padh) * (1.0 / this->ratio);
			int xmax = int(pdata[n * nout + 2] - this->Padw) * (1.0 / this->ratio);
			int ymax = int(pdata[n * nout + 3] - this->Padh) * (1.0 / this->ratio);
			int label = int(pdata[n * nout + 5]);
			// printf("Class: %s result is %d %d %d %d %f\n", class_names[label], xmin, ymin, xmax, ymax, box_score);
			char text[256];
			sprintf(text, "%s:%.3f", class_names[label], box_score);
			putText(frame, text, Point(xmin, ymin - 5), FONT_HERSHEY_TRIPLEX, 0.75, Scalar(0, 255, 0), 1);
			rectangle(frame, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255), 2);
		}
	}
#if use_camera
    cv::imshow("Fourcc", frame);
    cv::waitKey(1);
#else
	imwrite("result.jpg", frame);
#endif
}

int main(int argc, char **argv)
{
	Net_config v5Lite_cfg = {0.5, "../models/v5lite-e_end2end.onnx"};
	v5Lite net(v5Lite_cfg);

#if use_camera
    cv::VideoCapture capture;
    capture.open(0);

    cv::Mat frame;
    while (true)
    {
		struct timespec begin, end;
        long time;
        clock_gettime(CLOCK_MONOTONIC, &begin);

        capture >> frame;

        net.detect(frame, v5Lite_cfg);

        clock_gettime(CLOCK_MONOTONIC, &end);
        time = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec);
        if(time > 0) printf(">> Time : %lf ms\n", (double)time / 1000000);
    }
#else
	for (int i = 0; i < 100; i++)
	{
		struct timespec begin, end;
		long time;
		clock_gettime(CLOCK_MONOTONIC, &begin);

		string imgpath = "../images/000000001000.jpg";
		Mat srcimg = imread(imgpath);

		net.detect(srcimg, v5Lite_cfg);

		clock_gettime(CLOCK_MONOTONIC, &end);
		time = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec);
		if (time > 0) printf(">> Time : %lf ms\n", (double)time / 1000000);
	}
#endif

	return 0;
}
