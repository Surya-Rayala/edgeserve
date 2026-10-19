#pragma once

#include "runtime/llmRuntimeUtils.h"

#include <chrono>
#include <thread>

namespace trt_edgellm
{
namespace rt
{

inline std::optional<StreamChunk> StreamChannel::tryPop()
{
    bool stall = false;
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (!mPending.empty())
        {
            StreamChunk chunk = std::move(mPending.front());
            mPending.pop_front();
            return chunk;
        }
        // A drain pass ends at its first empty pop, so the second one after arming starts a pass.
        if (mHandshakeArmed && !mEmptyPopSeen && ++mEmptyPopsArmed == 2)
        {
            mEmptyPopSeen = true;
            stall = true;
        }
    }
    if (stall)
    {
        // Long enough for the producer to publish its last chunk and for handleRequest to return.
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
    return std::nullopt;
}

inline void StreamChannel::push(StreamChunk chunk)
{
    std::lock_guard<std::mutex> lock(mMutex);
    mPending.push_back(std::move(chunk));
}

inline void StreamChannel::finish(FinishReason)
{
    mFinished.store(true);
}

inline void StreamChannel::armEmptyPopHandshake()
{
    std::lock_guard<std::mutex> lock(mMutex);
    mHandshakeArmed = true;
}

inline bool StreamChannel::waitEmptyPop(int timeoutMs)
{
    // Polled rather than a timed condition wait, which older ThreadSanitizer runtimes do not intercept.
    for (int waited = 0; waited < timeoutMs; ++waited)
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            if (mEmptyPopSeen)
            {
                return true;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

} // namespace rt
} // namespace trt_edgellm
