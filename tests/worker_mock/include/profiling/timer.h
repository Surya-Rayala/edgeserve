#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace trt_edgellm
{
inline thread_local bool gMockProfilingEnabled = false;

inline void setProfilingEnabled(bool enabled) noexcept
{
    gMockProfilingEnabled = enabled;
}

inline bool getProfilingEnabled() noexcept
{
    return gMockProfilingEnabled;
}

namespace timer
{
struct StageTimingData
{
    std::vector<float> gpuTimesMs;
};

class Timer
{
public:
    void reset() noexcept
    {
        mData.clear();
    }

    //! Like a TIME_STAGE scope: recorded only while profiling is enabled on this thread.
    void record(std::string const& stage, float ms)
    {
        if (getProfilingEnabled())
        {
            mData[stage].gpuTimesMs.push_back(ms);
        }
    }

    std::optional<StageTimingData> getTimingData(std::string const& stage) const
    {
        auto const it = mData.find(stage);
        return it != mData.end() ? std::make_optional(it->second) : std::nullopt;
    }

private:
    std::map<std::string, StageTimingData> mData;
};
} // namespace timer

inline timer::Timer gTimer;
} // namespace trt_edgellm
