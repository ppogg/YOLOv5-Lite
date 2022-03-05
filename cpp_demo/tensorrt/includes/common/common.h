//
// Created by linghu8812 on 2021/2/8.
//

#ifndef LENET_TRT_COMMON_H
#define LENET_TRT_COMMON_H

#include <cuda_runtime_api.h>
#include <numeric>
#include <fstream>
#include <dirent.h>
#include "NvOnnxParser.h"
#include "logging.h"
#include <map>

// These is necessary if we want to be able to write 1_GiB instead of 1.0_GiB.
// Since the return type is signed, -1_GiB will work as expected.
constexpr long long int operator"" _GiB(long long unsigned int val)
{
    return val * (1 << 30);
}
constexpr long long int operator"" _MiB(long long unsigned int val)
{
    return val * (1 << 20);
}
constexpr long long int operator"" _KiB(long long unsigned int val)
{
    return val * (1 << 10);
}

static Logger gLogger{Logger::Severity::kINFO};
static LogStreamConsumer gLogVerbose{LOG_VERBOSE(gLogger)};
static LogStreamConsumer gLogInfo{LOG_INFO(gLogger)};
static LogStreamConsumer gLogWarning{LOG_WARN(gLogger)};
static LogStreamConsumer gLogError{LOG_ERROR(gLogger)};
static LogStreamConsumer gLogFatal{LOG_FATAL(gLogger)};

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

void setReportableSeverity(Logger::Severity severity);
std::vector<std::string>readFolder(const std::string &image_path);
std::map<int, std::string> readImageNetLabel(const std::string &fileName);
std::map<int, std::string> readCOCOLabel(const std::string &fileName);

#endif //LENET_TRT_COMMON_H
