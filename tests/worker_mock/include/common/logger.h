#pragma once

#include <NvInferRuntime.h>

namespace trt_edgellm
{
class Logger
{
public:
    void setLevel(nvinfer1::ILogger::Severity) noexcept {}
};

inline Logger gLogger;
} // namespace trt_edgellm
