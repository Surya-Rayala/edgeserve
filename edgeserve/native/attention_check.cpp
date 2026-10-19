/*
 * attention-check: compares the attention kernels the engines use, context phase (prefill, FMHA-v2) and
 * generation phase (decode, XQA), with a CPU reference on the paged KV layout, for the model shape below and
 * sequence lengths up to the engine's KV capacity. Diagnostic only: edgeserve setup runs it after every build.
 *
 *   ~/.local/share/edgeserve/native/current/attention-check [num_q_heads num_kv_heads head_dim kv_capacity]
 */

#include "common/tensor.h"
#include "kernels/contextAttentionKernels/cuteDslFMHAV2Runner.h"
#include "kernels/decodeAttentionKernels/decoderXQAJitCompiler.h"
#include "kernels/decodeAttentionKernels/decoderXQARunner.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

using namespace trt_edgellm;
namespace rt = trt_edgellm::rt;

namespace
{
constexpr int32_t kTokensPerPage = 128;

#define CHECK_CUDA(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t const e = (call);                                                                                  \
        if (e != cudaSuccess)                                                                                          \
        {                                                                                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__);               \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

struct Shape
{
    int32_t qHeads, kvHeads, dim, capacity, sm;
};

size_t bshd(int32_t seq, int32_t head, int32_t dim, int32_t heads, int32_t headDim)
{
    return (static_cast<size_t>(seq) * heads + head) * headDim + dim;
}

//! Runs the paged FMHA-v2 kernel exactly as the attention plugin does for a normal prefill (decode: the XQA kernel for
//! the last position) and checks query rows against a float reference. Returns the largest absolute error.
float maxError(Shape const& s, int32_t seqLen, bool sink, uint32_t seed, bool decode)
{
    int32_t const pages = (s.capacity + kTokensPerPage - 1) / kTokensPerPage; // pages per sequence (batch 1)
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uni(-1.0F, 1.0F);
    float const scale = 1.0F / std::sqrt(static_cast<float>(s.dim));

    std::vector<half> q(static_cast<size_t>(seqLen) * s.qHeads * s.dim);
    std::vector<half> k(static_cast<size_t>(seqLen) * s.kvHeads * s.dim);
    std::vector<half> v(k.size());
    if (sink)
    {
        // Every query is u (entries +-1); keys 0-15 are b*u, so their score is scale*b*dim = 10, while all other keys
        // are small noise (score ~0). Values: +1 for the sink keys, -1 elsewhere. While the sink keys are visible the
        // output is ~+0.98 for every query; a kernel that drops keys far back gives ~-1.
        std::vector<float> u(s.dim);
        for (auto& x : u)
        {
            x = uni(rng) > 0 ? 1.0F : -1.0F;
        }
        float const b = 10.0F / (scale * static_cast<float>(s.dim));
        for (int32_t t = 0; t < seqLen; ++t)
        {
            for (int32_t h = 0; h < s.qHeads; ++h)
            {
                for (int32_t d = 0; d < s.dim; ++d)
                {
                    q[bshd(t, h, d, s.qHeads, s.dim)] = __float2half(u[d]);
                }
            }
            bool const isSink = t < 16;
            for (int32_t h = 0; h < s.kvHeads; ++h)
            {
                for (int32_t d = 0; d < s.dim; ++d)
                {
                    k[bshd(t, h, d, s.kvHeads, s.dim)] = __float2half(isSink ? b * u[d] : 0.05F * uni(rng));
                    v[bshd(t, h, d, s.kvHeads, s.dim)] = __float2half(isSink ? 1.0F : -1.0F);
                }
            }
        }
    }
    else
    {
        for (auto& x : q)
        {
            x = __float2half(uni(rng));
        }
        for (auto& x : k)
        {
            x = __float2half(uni(rng));
        }
        for (auto& x : v)
        {
            x = __float2half(uni(rng));
        }
    }

    // Pool [2 * pages, 128, kvHeads, dim]: K pages in the first plane, V pages in the second, scrambled order.
    int32_t const flatPages = 2 * pages;
    std::vector<half> pool(static_cast<size_t>(flatPages) * kTokensPerPage * s.kvHeads * s.dim, __float2half(0.0F));
    std::vector<int32_t> pageList(static_cast<size_t>(2) * pages, -1);
    std::vector<int32_t> vOrder(pages);
    std::iota(vOrder.begin(), vOrder.end(), 0);
    std::shuffle(vOrder.begin(), vOrder.end(), std::mt19937(seed)); // any permutation: pages are looked up
    for (int32_t p = 0; p < pages; ++p)
    {
        pageList[p] = pages - 1 - p;          // K, reversed
        pageList[pages + p] = pages + vOrder[p]; // V, scrambled
    }
    for (int32_t t = 0; t < seqLen; ++t)
    {
        int32_t const kp = pageList[t / kTokensPerPage];
        int32_t const vp = pageList[pages + t / kTokensPerPage];
        for (int32_t h = 0; h < s.kvHeads; ++h)
        {
            for (int32_t d = 0; d < s.dim; ++d)
            {
                size_t const in = bshd(t, h, d, s.kvHeads, s.dim);
                size_t const tok = static_cast<size_t>(t % kTokensPerPage);
                pool[((static_cast<size_t>(kp) * kTokensPerPage + tok) * s.kvHeads + h) * s.dim + d] = k[in];
                pool[((static_cast<size_t>(vp) * kTokensPerPage + tok) * s.kvHeads + h) * s.dim + d] = v[in];
            }
        }
    }

    rt::Tensor qT({1, seqLen, s.qHeads, s.dim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor poolT({flatPages, kTokensPerPage, s.kvHeads, s.dim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor pageT({1, 2, pages}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor outT({1, seqLen, s.qHeads, s.dim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor cuQ({2}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor cuKV({2}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    int32_t const lens[2] = {0, seqLen};
    CHECK_CUDA(cudaMemcpy(qT.rawPointer(), q.data(), q.size() * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(poolT.rawPointer(), pool.data(), pool.size() * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(pageT.rawPointer(), pageList.data(), pageList.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(cuQ.rawPointer(), lens, sizeof(lens), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(cuKV.rawPointer(), lens, sizeof(lens), cudaMemcpyHostToDevice));

    std::vector<half> out(q.size());
    std::vector<int32_t> rows; // query positions checked against the reference
    if (decode)
    {
        // One generation step: the query of the last position attends to all seqLen keys in the cache.
        rt::Tensor lenT({1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
        CHECK_CUDA(cudaMemcpy(lenT.rawPointer(), &seqLen, sizeof(int32_t), cudaMemcpyHostToDevice));
        rt::Tensor q1({1, 1, s.qHeads, s.dim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::Tensor o1({1, 1, s.qHeads, s.dim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        size_t const row = static_cast<size_t>(s.qHeads) * s.dim;
        CHECK_CUDA(cudaMemcpy(q1.rawPointer(), q.data() + (seqLen - 1) * row, row * sizeof(half),
            cudaMemcpyHostToDevice));
        DecoderXQARunner runner(nvinfer1::DataType::kHALF, nvinfer1::DataType::kHALF, 1, s.qHeads, s.kvHeads, s.dim,
            s.sm);
        XQALaunchParams params = runner.initXQAParams();
        params.attentionScale = scale;
        params.output = o1.rawPointer();
        params.qInputPtr = q1.rawPointer();
        params.kvCache.data = poolT.rawPointer();
        params.kvCache.sequence_lengths = lenT.dataPointer<int32_t>();
        params.kvCache.capacity = static_cast<uint32_t>(pages * kTokensPerPage);
        params.kvCache.pageList = pageT.dataPointer<int32_t>();
        params.kvCache.tokensPerPage = kTokensPerPage;
        runner.dispatchXQAKernel(params, nullptr);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(out.data() + (seqLen - 1) * row, o1.rawPointer(), row * sizeof(half),
            cudaMemcpyDeviceToHost));
        rows.push_back(seqLen - 1);
    }
    else
    {
        CuteDslFMHAV2Runner runner(s.qHeads, s.kvHeads, s.dim, 1, seqLen, pages * kTokensPerPage);
        if (!runner.runPaged(qT.rawPointer(), poolT.rawPointer(), pageT.dataPointer<int32_t>(), outT.rawPointer(),
                cuQ.dataPointer<int32_t>(), cuKV.dataPointer<int32_t>(), flatPages, pages, kTokensPerPage, nullptr,
                scale, INT_MAX))
        {
            std::fprintf(stderr, "runPaged refused seqLen=%d\n", seqLen);
            std::exit(1);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(out.data(), outT.rawPointer(), out.size() * sizeof(half), cudaMemcpyDeviceToHost));
        // Sampled rows: the last 16 queries and every 97th one.
        for (int32_t t = 0; t < seqLen; t += 97)
        {
            rows.push_back(t);
        }
        for (int32_t t = std::max(0, seqLen - 16); t < seqLen; ++t)
        {
            rows.push_back(t);
        }
    }
    float maxErr = 0.0F;
    int32_t const group = s.qHeads / s.kvHeads;
    std::vector<float> w(seqLen);
    for (int32_t t : rows)
    {
        for (int32_t h = 0; h < s.qHeads; ++h)
        {
            int32_t const kvh = h / group;
            float m = -INFINITY;
            for (int32_t j = 0; j <= t; ++j)
            {
                float dot = 0.0F;
                for (int32_t d = 0; d < s.dim; ++d)
                {
                    dot += __half2float(q[bshd(t, h, d, s.qHeads, s.dim)]) * __half2float(k[bshd(j, kvh, d, s.kvHeads, s.dim)]);
                }
                w[j] = dot * scale;
                m = std::max(m, w[j]);
            }
            float sum = 0.0F;
            for (int32_t j = 0; j <= t; ++j)
            {
                w[j] = std::exp(w[j] - m);
                sum += w[j];
            }
            for (int32_t d = 0; d < s.dim; ++d)
            {
                float ref = 0.0F;
                for (int32_t j = 0; j <= t; ++j)
                {
                    ref += w[j] * __half2float(v[bshd(j, kvh, d, s.kvHeads, s.dim)]);
                }
                ref /= sum;
                float const got = __half2float(out[bshd(t, h, d, s.qHeads, s.dim)]);
                float const err = std::fabs(got - ref);
                if (!(err <= maxErr))
                {
                    maxErr = std::isnan(err) ? INFINITY : err;
                }
            }
        }
    }
    return maxErr;
}


//! Runs both kernels for every length the engine can see; returns the process exit status.
int run(Shape s)
{
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    s.sm = prop.major * 10 + prop.minor;
    // The decode kernel is compiled at run time (NVRTC), as llm_build does for an engine.
    XQAJitKey const key{s.sm, nvinfer1::DataType::kHALF, nvinfer1::DataType::kHALF, s.dim, s.qHeads / s.kvHeads,
        kTokensPerPage, false, false, false};
    XQAJitResult const jit = compileXQAKernel(key);
    if (!DecoderXQARunner::loadDecodeXQAKernelFromCubin(key, jit.cubin.data(), jit.cubin.size()))
    {
        std::fprintf(stderr, "could not load the XQA kernel\n");
        return 1;
    }
    std::printf("attention kernels vs CPU reference: q_heads=%d kv_heads=%d head_dim=%d kv_capacity=%d sm_%d\n",
        s.qHeads, s.kvHeads, s.dim, s.capacity, s.sm);
    int failures = 0;
    for (bool decode : {false, true})
    {
        for (int32_t len : {256, 1024, 2047, 2048, 2049, 2100, 2176, 2560, 3072, 3711})
        {
            if (len > s.capacity || (!decode && len > 3072))
            {
                continue;
            }
            float const random = maxError(s, len, false, 1234U + len, decode);
            float const sink = maxError(s, len, true, 99U + len, decode);
            bool const ok = random < 2e-2F && sink < 2e-2F;
            failures += ok ? 0 : 1;
            std::printf("  %s seq %5d: max |error| random %.5f, sink %.5f  %s\n", decode ? "decode " : "prefill", len,
                random, sink, ok ? "ok" : "MISMATCH");
            std::fflush(stdout);
        }
    }
    std::printf(failures ? "FAILED\n" : "all ok\n");
    return failures ? 1 : 0;
}
} // namespace

int main(int argc, char** argv)
{
    Shape s{32, 8, 128, 3712, 0}; // the bundled Llama-3.1 export and the default engine limits
    if (argc == 5)
    {
        s = {std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]), 0};
    }
    else if (argc != 1)
    {
        std::fprintf(stderr, "usage: %s [num_q_heads num_kv_heads head_dim kv_capacity]\n", argv[0]);
        return 2;
    }
    if (s.qHeads <= 0 || s.kvHeads <= 0 || s.dim <= 0 || s.capacity <= 0 || s.qHeads % s.kvHeads != 0)
    {
        std::fprintf(stderr, "attention-check: the sizes must be positive, num_q_heads a multiple of num_kv_heads\n");
        return 2;
    }
    try
    {
        return run(s);
    }
    catch (std::exception const& e)
    {
        std::fprintf(stderr, "attention-check: %s\n", e.what());
        return 1;
    }
}
