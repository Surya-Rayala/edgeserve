#pragma once

#include <cstdlib>
#include <memory>

namespace trt_edgellm
{
struct DlDeleter
{
    void operator()(void*) const noexcept {}
};

//! $MOCK_NO_PLUGIN simulates a missing plugin library.
inline std::unique_ptr<void, DlDeleter> loadEdgellmPluginLib() noexcept
{
    static int handle;
    return std::unique_ptr<void, DlDeleter>(std::getenv("MOCK_NO_PLUGIN") != nullptr ? nullptr : &handle);
}
} // namespace trt_edgellm
