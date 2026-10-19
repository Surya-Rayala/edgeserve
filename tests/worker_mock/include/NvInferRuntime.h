#pragma once

namespace nvinfer1
{
class ILogger
{
public:
    enum class Severity : int
    {
        kINTERNAL_ERROR = 0,
        kERROR = 1,
        kWARNING = 2,
        kINFO = 3,
        kVERBOSE = 4,
    };
};
} // namespace nvinfer1

inline int getInferLibVersion() noexcept
{
    return 100300;
}
