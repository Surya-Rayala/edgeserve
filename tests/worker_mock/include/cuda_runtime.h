// Stand-in for the CUDA runtime calls native/worker.cpp makes; nothing touches a GPU.
#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <thread>

typedef struct CUstream_st* cudaStream_t;

enum cudaError_t
{
    cudaSuccess = 0,
    cudaErrorMemoryAllocation = 2,
    cudaErrorNoDevice = 100,
    cudaErrorIllegalAddress = 700,
};

enum cudaDeviceAttr
{
    cudaDevAttrComputeCapabilityMajor,
    cudaDevAttrComputeCapabilityMinor,
    cudaDevAttrMultiProcessorCount,
    cudaDevAttrIntegrated,
    cudaDevAttrGlobalMemoryBusWidth,
    cudaDevAttrL2CacheSize,
    cudaDevAttrClockRate,
    cudaDevAttrMemoryClockRate,
};

struct cudaDeviceProp
{
    char name[256];
};

#define cudaStreamNonBlocking 0x01

namespace mockcuda
{
//! Per host thread as in the real runtime: set by a failing call, cleared only by cudaGetLastError().
inline thread_local cudaError_t lastError = cudaSuccess;
//! Once set, every synchronize fails (an unusable context).
inline std::atomic<int> stickyError{cudaSuccess};
} // namespace mockcuda

inline cudaError_t cudaGetLastError()
{
    cudaError_t const error = mockcuda::lastError;
    mockcuda::lastError = cudaSuccess;
    return error;
}

inline char const* cudaGetErrorString(cudaError_t error)
{
    switch (error)
    {
    case cudaSuccess: return "no error";
    case cudaErrorMemoryAllocation: return "out of memory";
    case cudaErrorNoDevice: return "no CUDA-capable device is detected";
    case cudaErrorIllegalAddress: return "an illegal memory access was encountered";
    }
    return "unknown error";
}

inline cudaError_t cudaDeviceSynchronize()
{
    auto const sticky = static_cast<cudaError_t>(mockcuda::stickyError.load());
    if (sticky != cudaSuccess)
    {
        mockcuda::lastError = sticky;
    }
    return sticky;
}

inline cudaError_t cudaFree(void*)
{
    return cudaSuccess;
}

//! $MOCK_MEMINFO_MS makes the query slow.
inline cudaError_t cudaMemGetInfo(size_t* freeBytes, size_t* totalBytes)
{
    if (char const* ms = std::getenv("MOCK_MEMINFO_MS"))
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(std::atol(ms)));
    }
    *freeBytes = size_t{30} << 30;
    *totalBytes = size_t{64} << 30;
    return cudaSuccess;
}

inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int)
{
    static int dummy;
    *stream = reinterpret_cast<cudaStream_t>(&dummy);
    return cudaSuccess;
}

inline cudaError_t cudaStreamDestroy(cudaStream_t)
{
    return cudaSuccess;
}

//! $MOCK_NO_DEVICE simulates a machine without a usable GPU.
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* props, int)
{
    if (std::getenv("MOCK_NO_DEVICE") != nullptr)
    {
        mockcuda::lastError = cudaErrorNoDevice;
        return cudaErrorNoDevice;
    }
    std::strcpy(props->name, "MockGPU");
    return cudaSuccess;
}

inline cudaError_t cudaRuntimeGetVersion(int* version)
{
    *version = 12060;
    return cudaSuccess;
}

inline cudaError_t cudaDriverGetVersion(int* version)
{
    *version = 12060;
    return cudaSuccess;
}

inline cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int)
{
    switch (attr)
    {
    case cudaDevAttrComputeCapabilityMajor: *value = 8; break;
    case cudaDevAttrComputeCapabilityMinor: *value = 7; break;
    case cudaDevAttrMultiProcessorCount: *value = 8; break;
    case cudaDevAttrIntegrated: *value = 1; break;
    case cudaDevAttrGlobalMemoryBusWidth: *value = 128; break;
    case cudaDevAttrL2CacheSize: *value = 2 << 20; break;
    case cudaDevAttrClockRate: *value = 1020000; break;
    case cudaDevAttrMemoryClockRate: *value = 3199000; break;
    }
    return cudaSuccess;
}
