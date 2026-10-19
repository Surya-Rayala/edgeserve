/*
 * edgeserve-worker: owns the GPU and runs one TensorRT Edge-LLM engine for the edgeserve API.
 *
 *   edgeserve-worker --probe
 *       Print one JSON line with GPU / CUDA / TensorRT / Edge-LLM facts and exit.
 *   edgeserve-worker --engine-dir DIR [--no-cuda-graph] [--verbose]
 *       Load the engine, warm it up, then serve commands until stdin closes.
 *
 * Protocol: one JSON object per line. Commands arrive on stdin and events leave on stdout;
 * every log line (including Edge-LLM's own INFO output) goes to stderr. Message fields are
 * documented in docs/ARCHITECTURE.md ("Worker protocol").
 *
 * Timing: the per-token callback fires on the decode thread right after each step's stream
 * synchronize, so its timestamps are the moments tokens became available. With static
 * batching every sequence in a batch advances in lockstep, so the worker records one
 * timestamp per batch step and sequence r's i-th token belongs to step i. GPU time per step
 * comes from Edge-LLM's CUDA-event stage timers.
 */

#include "common/logger.h"
#include "common/trtUtils.h"
#include "common/version.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/streaming.h"
#include "tokenizer/tokenizer.h"

#include <NvInferRuntime.h>
#include <NvInferVersion.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace rt = trt_edgellm::rt;
using json = nlohmann::json;

namespace
{

//! Longest command line accepted; longer lines are skipped. Twice the largest command the
//! front-end sends (a generate batches up to 16 requests whose bodies are capped at 4 MiB), it
//! only bounds memory on a misbehaving sender: a skipped generate never gets a terminal event.
constexpr size_t kMaxCommandBytes = size_t{128} << 20;

//! Largest top_k passed to Edge-LLM (the front-end turns a top_k covering the whole vocabulary into "no limit").
constexpr int64_t kMaxTopK = 1024;

int64_t nowNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

//! Length of the longest prefix of `text` that does not end inside a multi-byte UTF-8 character. Edge-LLM's
//! stream chunks can cut a character in two (its stop-string look-ahead holds back whole bytes); the next chunk
//! completes it. Invalid bytes are not held back.
size_t completeUtf8Prefix(std::string const& text)
{
    size_t const end = text.size();
    size_t start = end; // start of the continuation bytes at the end
    while (start > 0 && end - start < 3 && (static_cast<unsigned char>(text[start - 1]) & 0xC0) == 0x80)
    {
        --start;
    }
    if (start == 0)
    {
        return end;
    }
    unsigned char const lead = static_cast<unsigned char>(text[start - 1]);
    size_t length = 0; // of the character this byte starts; 0: not the lead byte of a multi-byte character
    if (lead >= 0xC2 && lead <= 0xDF)
    {
        length = 2;
    }
    else if (lead >= 0xE0 && lead <= 0xEF)
    {
        length = 3;
    }
    else if (lead >= 0xF0 && lead <= 0xF4)
    {
        length = 4;
    }
    return length > end - (start - 1) ? start - 1 : end;
}

void logLine(char const* level, std::string const& message)
{
    std::cerr << "[edgeserve-worker] " << level << ": " << message << std::endl;
}

json errorEvent(json const& id, std::string const& code, std::string const& message, bool fatal)
{
    return {{"ev", "error"}, {"id", id}, {"code", code}, {"message", message}, {"fatal", fatal}};
}

//! Serialises events onto the protocol fd from its own thread so the decode loop never blocks
//! on a slow reader.
class EventWriter
{
public:
    explicit EventWriter(int fd)
        : mFd(fd)
        , mThread([this] { run(); })
    {
    }

    ~EventWriter()
    {
        close();
    }

    void push(json const& event)
    {
        std::string line = event.dump(-1, ' ', false, json::error_handler_t::replace);
        line.push_back('\n');
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mQueue.push_back(std::move(line));
        }
        mCv.notify_one();
    }

    //! Flushes everything queued so far, then stops the writer thread.
    void close()
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            if (mClosed)
            {
                return;
            }
            mClosed = true;
        }
        mCv.notify_one();
        if (mThread.joinable())
        {
            mThread.join();
        }
    }

private:
    void run()
    {
        while (true)
        {
            std::deque<std::string> batch;
            {
                std::unique_lock<std::mutex> lock(mMutex);
                mCv.wait(lock, [this] { return mClosed || !mQueue.empty(); });
                if (mQueue.empty())
                {
                    return; // closed and drained
                }
                batch.swap(mQueue);
            }
            for (auto const& line : batch)
            {
                writeAll(line);
            }
        }
    }

    void writeAll(std::string const& data)
    {
        size_t offset = 0;
        while (offset < data.size())
        {
            ssize_t const written = ::write(mFd, data.data() + offset, data.size() - offset);
            if (written < 0)
            {
                if (errno == EINTR)
                {
                    continue;
                }
                std::_Exit(0); // the front-end is gone; nobody is left to serve
            }
            offset += static_cast<size_t>(written);
        }
    }

    int mFd;
    std::mutex mMutex;
    std::condition_variable mCv;
    std::deque<std::string> mQueue;
    bool mClosed{false};
    std::thread mThread; // declared last: starts once the members above exist
};

//! Splits a byte stream into newline-terminated lines without letting one line grow without bound.
class LineReader
{
public:
    LineReader(int fd, size_t maxLineBytes)
        : mFd(fd)
        , mMaxLineBytes(maxLineBytes)
    {
    }

    //! Next line (without its newline) in `line`; false at end of input. A line longer than the
    //! limit is consumed and dropped as it arrives: it comes back empty with `tooLong` set.
    bool next(std::string& line, bool& tooLong)
    {
        line.clear();
        tooLong = false;
        while (true)
        {
            size_t const newline = mBuffer.find('\n', mScanned);
            if (newline != std::string::npos)
            {
                if (!tooLong)
                {
                    line.assign(mBuffer, mStart, newline - mStart);
                }
                mStart = newline + 1;
                mScanned = mStart;
                return true;
            }
            if (mBuffer.size() - mStart > mMaxLineBytes)
            {
                tooLong = true;
                mBuffer.clear();
            }
            else
            {
                mBuffer.erase(0, mStart);
            }
            mStart = 0;
            mScanned = mBuffer.size();
            if (mEnd)
            {
                // A last line without a newline still counts.
                bool const pending = tooLong || !mBuffer.empty();
                if (!tooLong)
                {
                    line.swap(mBuffer);
                }
                mBuffer.clear();
                mScanned = 0;
                return pending;
            }
            char chunk[1 << 16];
            ssize_t const got = ::read(mFd, chunk, sizeof(chunk));
            if (got > 0)
            {
                mBuffer.append(chunk, static_cast<size_t>(got));
            }
            else if (got == 0 || errno != EINTR)
            {
                mEnd = true;
            }
        }
    }

private:
    int mFd;
    size_t mMaxLineBytes;
    std::string mBuffer;
    size_t mStart{0};   //!< first byte of the current line in mBuffer
    size_t mScanned{0}; //!< bytes of mBuffer already searched for a newline
    bool mEnd{false};
};

std::string cudaVersionString(int version)
{
    return std::to_string(version / 1000) + "." + std::to_string((version % 1000) / 10);
}

int deviceAttribute(cudaDeviceAttr attr)
{
    int value = 0;
    cudaDeviceGetAttribute(&value, attr, 0);
    return value;
}

json deviceFacts()
{
    cudaDeviceProp props{};
    cudaError_t const status = cudaGetDeviceProperties(&props, 0);
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string("no usable CUDA device: ") + cudaGetErrorString(status));
    }
    int runtimeVersion = 0;
    int driverVersion = 0;
    cudaRuntimeGetVersion(&runtimeVersion);
    cudaDriverGetVersion(&driverVersion);
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    cudaMemGetInfo(&freeBytes, &totalBytes);
    int const trtRuntime = getInferLibVersion();

    return json{
        {"gpu_name", props.name},
        {"sm", deviceAttribute(cudaDevAttrComputeCapabilityMajor) * 10
                + deviceAttribute(cudaDevAttrComputeCapabilityMinor)},
        {"sm_count", deviceAttribute(cudaDevAttrMultiProcessorCount)},
        {"integrated", deviceAttribute(cudaDevAttrIntegrated) != 0},
        {"total_memory_bytes", totalBytes},
        {"free_memory_bytes", freeBytes},
        {"memory_bus_width_bits", deviceAttribute(cudaDevAttrGlobalMemoryBusWidth)},
        {"l2_cache_bytes", deviceAttribute(cudaDevAttrL2CacheSize)},
        {"clock_rate_khz", deviceAttribute(cudaDevAttrClockRate)},
        {"memory_clock_rate_khz", deviceAttribute(cudaDevAttrMemoryClockRate)},
        {"cuda_runtime", cudaVersionString(runtimeVersion)},
        {"cuda_driver", cudaVersionString(driverVersion)},
        {"tensorrt",
            std::to_string(NV_TENSORRT_MAJOR) + "." + std::to_string(NV_TENSORRT_MINOR) + "."
                + std::to_string(NV_TENSORRT_PATCH) + "." + std::to_string(NV_TENSORRT_BUILD)},
        {"tensorrt_runtime",
            std::to_string(trtRuntime / 10000) + "." + std::to_string((trtRuntime / 100) % 100) + "."
                + std::to_string(trtRuntime % 100)},
        {"edgellm", trt_edgellm::version::kRUNTIME_VERSION},
    };
}

std::string finishReasonString(rt::FinishReason reason)
{
    switch (reason)
    {
    case rt::FinishReason::kEndId:
    case rt::FinishReason::kStopWords: return "stop";
    case rt::FinishReason::kLength: return "length";
    case rt::FinishReason::kCancelled: return "cancelled";
    case rt::FinishReason::kError: return "error";
    case rt::FinishReason::kNotFinished: break;
    }
    return "error";
}

//! Raised for malformed commands; reported to the front-end, never fatal.
struct BadRequest : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

//! An integer field within [lo, hi], or `fallback` when it is absent.
int64_t integerField(json const& object, char const* key, int64_t fallback, int64_t lo, int64_t hi)
{
    auto const it = object.find(key);
    if (it == object.end())
    {
        return fallback;
    }
    if (it->is_number_integer()
        && (!it->is_number_unsigned()
            || it->get<uint64_t>() <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())))
    {
        int64_t const value = it->get<int64_t>();
        if (value >= lo && value <= hi)
        {
            return value;
        }
    }
    throw BadRequest(
        "'" + std::string(key) + "' must be an integer from " + std::to_string(lo) + " to " + std::to_string(hi));
}

/*!
 * Build an Edge-LLM request from a command's "requests" (+ "sampling") fields.
 * Each item is either {"messages": [...]} (rendered with the model's chat template) or
 * {"prompt": "..."} (already formatted; tokenized as-is, BOS added if the text lacks it).
 */
rt::LLMGenerationRequest buildRequest(json const& cmd, trt_edgellm::tokenizer::Tokenizer const& tokenizer)
{
    rt::LLMGenerationRequest request{};
    json const sampling = cmd.value("sampling", json::object());
    if (!sampling.is_object())
    {
        throw BadRequest("'sampling' must be an object");
    }
    request.temperature = sampling.value("temperature", 0.0F);
    request.topP = sampling.value("top_p", 1.0F);
    // The runtime narrows both to int32.
    // Edge-LLM's top-k kernels size shared memory and an int32 workspace by top_k (sampling.cu): bound it.
    request.topK = integerField(sampling, "top_k", 1, 0, kMaxTopK);
    // Edge-LLM needs top_k > 0 or top_p < 1, and samples at all only if top_k > 1, top_p < 1 - 1e-6 or the
    // temperature is not 1 (shouldUseNonGreedySampling, samplingUtils.cpp). "No limit" (top_k 0, top_p 1) is
    // therefore greedy at temperature 0 and otherwise the largest top_p it counts as set, which keeps all but a
    // ~1e-6 tail of the distribution.
    constexpr float kTopPUnset = 1.0F - 1e-6F;
    if (request.topK <= 0 && request.topP >= kTopPUnset)
    {
        if (request.temperature < 1e-3F)
        {
            request.topK = 1;
        }
        else
        {
            request.topP = std::nextafter(kTopPUnset, 0.0F);
        }
    }
    request.maxGenerateLength = integerField(sampling, "max_new_tokens", 1, 1, INT32_MAX);
    request.applyChatTemplate = true;
    request.addGenerationPrompt = true;
    request.skipSpecialTokens = true;

    json const& items = cmd.at("requests");
    if (!items.is_array() || items.empty())
    {
        throw BadRequest("'requests' must be a non-empty array");
    }
    bool anyPrompt = false;
    bool anyMessages = false;
    for (json const& item : items)
    {
        rt::LLMGenerationRequest::Request entry;
        if (item.contains("prompt"))
        {
            anyPrompt = true;
            auto ids = tokenizer.encode(item.at("prompt").get<std::string>(), /*addBos=*/false);
            auto const bos = tokenizer.getBosId();
            if (bos >= 0 && (ids.empty() || ids.front() != bos))
            {
                ids.insert(ids.begin(), bos);
            }
            request.preTokenizedInputIds.emplace_back(ids.begin(), ids.end());
        }
        else
        {
            anyMessages = true;
            for (json const& message : item.at("messages"))
            {
                rt::Message converted;
                converted.role = message.at("role").get<std::string>();
                converted.contents.push_back({"text", message.at("content").get<std::string>()});
                entry.messages.push_back(std::move(converted));
            }
            if (entry.messages.empty())
            {
                throw BadRequest("'messages' must not be empty");
            }
        }
        if (item.contains("stop") && !item["stop"].is_null())
        {
            entry.stopStrings = item["stop"].get<std::vector<std::string>>();
        }
        if (item.contains("seed") && !item["seed"].is_null())
        {
            if (!item["seed"].is_number_integer())
            {
                throw BadRequest("'seed' must be an integer");
            }
            entry.samplingSeed = item["seed"].get<uint64_t>();
        }
        request.requests.push_back(std::move(entry));
    }
    if (anyPrompt && anyMessages)
    {
        throw BadRequest("a batch must not mix 'prompt' and 'messages' requests");
    }
    return request;
}

//! True when the CUDA context is unusable (sticky error); the process must then exit. Otherwise
//! clears this thread's last error so that a non-sticky failure is not reported again by the
//! launch checks (cudaGetLastError) of the next request.
bool cudaContextBroken(std::string& message)
{
    cudaError_t const status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        message = cudaGetErrorString(status);
        // A non-sticky error clears on read; only a second failure means the context is gone.
        if (cudaDeviceSynchronize() != cudaSuccess)
        {
            return true;
        }
    }
    static_cast<void>(cudaGetLastError());
    return false;
}

class Worker
{
public:
    //! A generate command, registered when it arrives so that a cancel reaches it while it waits.
    struct Job
    {
        json cmd;
        int64_t receivedNs{0};
        json id;
        std::vector<std::shared_ptr<rt::StreamChannel>> channels; //!< one per sequence; cancel() stops it
    };

    Worker(std::string engineDir, bool captureCudaGraph, EventWriter& out)
        : mEngineDir(std::move(engineDir))
        , mCaptureCudaGraph(captureCudaGraph)
        , mOut(out)
    {
    }

    ~Worker()
    {
        mRuntime.reset();
        if (mStream != nullptr)
        {
            cudaStreamDestroy(mStream);
        }
    }

    //! Load the engine, capture decode CUDA graphs, warm up, then announce readiness.
    void load()
    {
        int64_t const t0 = nowNs();
        cudaFree(nullptr); // create the context first so the free-memory baseline excludes it
        size_t freeBefore = 0;
        size_t total = 0;
        cudaMemGetInfo(&freeBefore, &total);

        mPlugin = trt_edgellm::loadEdgellmPluginLib();
        if (!mPlugin)
        {
            throw std::runtime_error("could not load the Edge-LLM TensorRT plugin (EDGELLM_PLUGIN_PATH)");
        }
        if (!mTokenizer.loadFromHF(mEngineDir))
        {
            throw std::runtime_error("could not load the tokenizer from " + mEngineDir);
        }
        if (cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking) != cudaSuccess)
        {
            throw std::runtime_error("could not create a CUDA stream");
        }
        mRuntime = std::make_unique<rt::LLMInferenceRuntime>(
            mEngineDir, "", std::unordered_map<std::string, std::string>{}, mStream);
        int64_t const tLoaded = nowNs();

        bool graphCaptured = false;
        if (mCaptureCudaGraph)
        {
            graphCaptured = mRuntime->captureDecodingCUDAGraph(mStream);
            if (!graphCaptured)
            {
                logLine("warning", "CUDA graph capture failed; decoding without graphs");
            }
        }

        // Warm-up: loads lazily-initialised kernel modules so the first real request is not slower.
        int64_t const tWarm = nowNs();
        json warm = {{"requests", json::array({{{"messages", json::array({{{"role", "user"}, {"content", "Hi"}}})}}})},
            {"sampling", {{"temperature", 0.0}, {"top_p", 1.0}, {"top_k", 1}, {"max_new_tokens", 2}}}};
        rt::LLMGenerationRequest warmRequest = buildRequest(warm, mTokenizer);
        rt::LLMGenerationResponse warmResponse;
        if (!mRuntime->handleRequest(warmRequest, warmResponse, mStream))
        {
            throw std::runtime_error("warm-up generation failed");
        }
        int64_t const tReady = nowNs();

        size_t freeAfter = 0;
        cudaMemGetInfo(&freeAfter, &total);
        json ready = {{"ev", "ready"}, {"engine_dir", mEngineDir}, {"max_batch_size", mRuntime->maxBatchSize()},
            {"cuda_graph", graphCaptured}, {"load_ms", (tLoaded - t0) / 1e6}, {"warmup_ms", (tReady - tWarm) / 1e6},
            {"device_free_before_load_bytes", freeBefore}, {"device_free_after_load_bytes", freeAfter},
            {"device", deviceFacts()}};
        mOut.push(ready);
    }

    //! Prompt-token counts after chat templating; safe to call while a generation runs.
    void count(json const& cmd) const
    {
        try
        {
            rt::LLMGenerationRequest request = buildRequest(cmd, mTokenizer);
            std::vector<int32_t> counts;
            if (!request.preTokenizedInputIds.empty())
            {
                for (auto const& ids : request.preTokenizedInputIds)
                {
                    counts.push_back(static_cast<int32_t>(ids.size()));
                }
            }
            else
            {
                counts = mRuntime->countPromptTokens(request);
            }
            mOut.push({{"ev", "count"}, {"id", cmd.at("id")}, {"prompt_tokens", counts}});
        }
        catch (std::exception const& e)
        {
            mOut.push(errorEvent(cmd.value("id", json()), "bad_request", e.what(), false));
        }
    }

    //! Register a generate command (an object) as soon as it arrives.
    std::shared_ptr<Job> accept(json cmd, int64_t receivedNs)
    {
        auto job = std::make_shared<Job>();
        job->id = cmd.value("id", json());
        auto const items = cmd.find("requests");
        // Oversized or malformed batches get no channels; generate() rejects them.
        if (items != cmd.end() && items->is_array()
            && items->size() <= static_cast<size_t>(std::max(0, mRuntime->maxBatchSize())))
        {
            for (size_t i = 0; i < items->size(); ++i)
            {
                auto channel = rt::StreamChannel::create();
                channel->setStreamInterval(1);
                job->channels.push_back(channel);
            }
        }
        job->cmd = std::move(cmd);
        job->receivedNs = receivedNs;
        std::lock_guard<std::mutex> lock(mJobsMutex);
        mJobs.push_back(job);
        return job;
    }

    //! Cancel every accepted job (at shutdown, or when the front-end goes away).
    void cancelAll()
    {
        std::lock_guard<std::mutex> lock(mJobsMutex);
        for (auto const& job : mJobs)
        {
            for (auto const& channel : job->channels)
            {
                channel->cancel();
            }
        }
    }

    //! Cancel a queued or running job (all of its sequences, or one when "req" is given).
    void cancel(json const& cmd)
    {
        auto const id = cmd.find("id");
        if (id == cmd.end())
        {
            return;
        }
        std::optional<uint64_t> only;
        auto const req = cmd.find("req");
        if (req != cmd.end() && !req->is_null())
        {
            if (!req->is_number_integer())
            {
                throw BadRequest("'req' must be an integer");
            }
            if (req->is_number_unsigned())
            {
                only = req->get<uint64_t>(); // a negative index cancels the whole job
            }
        }
        std::lock_guard<std::mutex> lock(mJobsMutex);
        for (auto const& job : mJobs)
        {
            if (job->id != *id)
            {
                continue;
            }
            for (size_t i = 0; i < job->channels.size(); ++i)
            {
                if (!only || *only == i)
                {
                    job->channels[i]->cancel();
                }
            }
        }
    }

    //! Run one accepted generate command to completion and report exactly one terminal event for
    //! it ("done" or "error"). Returns false when the CUDA context is broken and the process must exit.
    bool generate(Job const& job)
    {
        bool contextUsable = true;
        json terminal;
        try
        {
            terminal = run(job, contextUsable);
        }
        catch (std::exception const& e)
        {
            terminal = errorEvent(job.id, "generation_failed", e.what(), false);
        }
        {
            std::lock_guard<std::mutex> lock(mJobsMutex);
            mJobs.erase(std::remove_if(mJobs.begin(), mJobs.end(), [&job](auto const& j) { return j.get() == &job; }),
                mJobs.end());
        }
        mOut.push(terminal);
        return contextUsable;
    }

private:
    struct StepLog
    {
        std::vector<int64_t> tNs;
        std::vector<uint64_t> freeBytes;
        std::vector<int32_t> active; //!< sequences that produced a token in the step
        int32_t lastBatchIdx{INT32_MAX};

        void reserve(size_t n)
        {
            tNs.reserve(n);
            freeBytes.reserve(n);
            active.reserve(n);
        }
    };

    //! The generation itself; returns the terminal event. Text events are pushed as they arrive.
    json run(Job const& job, bool& contextUsable)
    {
        json const& id = job.id;
        rt::LLMGenerationRequest request{};
        bool streamText = false;
        try
        {
            request = buildRequest(job.cmd, mTokenizer);
            auto const stream = job.cmd.find("stream");
            if (stream != job.cmd.end())
            {
                if (!stream->is_boolean())
                {
                    throw BadRequest("'stream' must be true or false");
                }
                streamText = stream->get<bool>();
            }
        }
        catch (std::exception const& e)
        {
            return errorEvent(id, "bad_request", e.what(), false);
        }
        size_t const n = request.requests.size();
        if (n > static_cast<size_t>(std::max(0, mRuntime->maxBatchSize())))
        {
            return errorEvent(id, "bad_request",
                "batch of " + std::to_string(n) + " exceeds the engine's max batch size "
                    + std::to_string(mRuntime->maxBatchSize()),
                false);
        }
        request.streamChannels = job.channels; // accept() made one per sequence

        // Per-step record: the first callback of each batch step marks when that step's tokens
        // became available (one stream synchronize serves the whole batch).
        StepLog steps;
        steps.reserve(static_cast<size_t>(std::min<int64_t>(request.maxGenerateLength, 1 << 16)) + 1);
        request.onTokenGenerated = [&steps](rt::TokenCallbackInfo const& info) {
            if (info.batchIdx <= steps.lastBatchIdx)
            {
                steps.tNs.push_back(nowNs()); // first, so the memory query's latency is not included
                size_t freeBytes = 0;
                size_t totalBytes = 0;
                cudaMemGetInfo(&freeBytes, &totalBytes);
                steps.freeBytes.push_back(freeBytes);
                steps.active.push_back(0);
            }
            steps.lastBatchIdx = info.batchIdx;
            steps.active.back() += 1;
        };

        std::atomic<bool> generationDone{false};
        std::thread textPump;
        if (streamText)
        {
            textPump = std::thread([this, &job, &generationDone] { pumpText(job, generationDone); });
        }

        trt_edgellm::gTimer.reset();
        trt_edgellm::setProfilingEnabled(true); // thread-local: records stage GPU times on this thread
        rt::LLMGenerationResponse response;
        bool ok = false;
        std::string failure;
        int64_t const startNs = nowNs();
        try
        {
            ok = mRuntime->handleRequest(request, response, mStream);
        }
        catch (std::exception const& e)
        {
            failure = e.what();
        }
        catch (...)
        {
            failure = "unknown exception in the runtime";
        }
        int64_t const endNs = nowNs();
        trt_edgellm::setProfilingEnabled(false);

        generationDone = true;
        if (textPump.joinable())
        {
            textPump.join();
        }

        std::string cudaMessage;
        if (cudaContextBroken(cudaMessage))
        {
            contextUsable = false;
            return errorEvent(id, "cuda_error", "CUDA context is unusable: " + cudaMessage, true);
        }
        if (!failure.empty() || !ok)
        {
            std::string const code
                = failure.find("EDGELLM_INPUT_TOO_LONG") != std::string::npos ? "input_too_long" : "generation_failed";
            if (failure.empty())
            {
                failure = "the runtime rejected the request (see worker log)";
            }
            return errorEvent(id, code, failure, false);
        }

        return doneEvent(id, job.receivedNs, startNs, endNs, steps, response, n);
    }

    //! Forward text deltas as soon as Edge-LLM publishes them (UTF-8 safe, stop strings held back).
    void pumpText(Job const& job, std::atomic<bool> const& generationDone)
    {
        std::vector<bool> finished(job.channels.size(), false);
        std::vector<std::string> pending(job.channels.size()); // the start of a character cut by a chunk boundary
        while (true)
        {
            // Read before draining: every chunk is pushed before the flag is set, so a drain that
            // starts after seeing it set collects the last ones too.
            bool const done = generationDone;
            bool progressed = false;
            for (size_t i = 0; i < job.channels.size(); ++i)
            {
                while (auto chunk = job.channels[i]->tryPop())
                {
                    progressed = true;
                    pending[i] += chunk->text;
                    size_t const ready = completeUtf8Prefix(pending[i]);
                    if (ready > 0)
                    {
                        mOut.push({{"ev", "text"}, {"id", job.id}, {"req", i}, {"t_ns", nowNs()},
                            {"text", pending[i].substr(0, ready)}});
                        pending[i].erase(0, ready);
                    }
                    finished[i] = finished[i] || chunk->finished;
                }
            }
            bool const allFinished = std::all_of(finished.begin(), finished.end(), [](bool f) { return f; });
            if (allFinished || done)
            {
                return;
            }
            if (!progressed)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
    }

    json doneEvent(json const& id, int64_t receivedNs, int64_t startNs, int64_t endNs, StepLog const& steps,
        rt::LLMGenerationResponse const& response, size_t n) const
    {
        std::vector<float> prefillGpu;
        std::vector<float> decodeGpu;
        if (auto data = trt_edgellm::gTimer.getTimingData(trt_edgellm::metrics::StageNames::kLLM_PREFILL))
        {
            prefillGpu = data->gpuTimesMs;
        }
        if (auto data = trt_edgellm::gTimer.getTimingData(trt_edgellm::metrics::StageNames::kLLM_GENERATION))
        {
            decodeGpu = data->gpuTimesMs;
        }
        json gpuMs = json::array();
        for (size_t step = 0; step < steps.tNs.size(); ++step)
        {
            if (step == 0)
            {
                gpuMs.push_back(prefillGpu.empty() ? json() : json(prefillGpu.front()));
            }
            else
            {
                gpuMs.push_back(step - 1 < decodeGpu.size() ? json(decodeGpu[step - 1]) : json());
            }
        }

        json results = json::array();
        for (size_t r = 0; r < n; ++r)
        {
            std::vector<int32_t> tokenIds = r < response.outputIds.size() ? response.outputIds[r] : std::vector<int32_t>{};
            json pieces = json::array();
            for (int32_t token : tokenIds)
            {
                pieces.push_back(mTokenizer.idToPiece(token, /*skipSpecialTokens=*/false));
            }
            results.push_back({{"req", r},
                {"finish_reason", r < response.finishReasons.size() ? finishReasonString(response.finishReasons[r])
                                                                     : std::string("error")},
                {"prompt_tokens", r < response.inputTokenCounts.size() ? response.inputTokenCounts[r] : -1},
                {"text", r < response.outputTexts.size() ? response.outputTexts[r] : std::string()},
                {"token_ids", tokenIds}, {"pieces", pieces}});
        }

        return {{"ev", "done"}, {"id", id}, {"t_received_ns", receivedNs}, {"t_start_ns", startNs},
            {"t_end_ns", endNs},
            {"steps", {{"t_ns", steps.tNs}, {"gpu_ms", gpuMs}, {"free_bytes", steps.freeBytes}, {"active", steps.active}}},
            {"results", results}};
    }

    std::string mEngineDir;
    bool mCaptureCudaGraph;
    EventWriter& mOut;
    std::unique_ptr<void, trt_edgellm::DlDeleter> mPlugin{nullptr};
    trt_edgellm::tokenizer::Tokenizer mTokenizer;
    cudaStream_t mStream{nullptr};
    std::unique_ptr<rt::LLMInferenceRuntime> mRuntime;
    std::mutex mJobsMutex;
    std::vector<std::shared_ptr<Job>> mJobs; //!< accepted and not yet finished, queued or running
};

struct Args
{
    bool probe{false};
    std::string engineDir;
    bool captureCudaGraph{true};
    bool verbose{false};
};

Args parseArgs(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc; ++i)
    {
        std::string const arg = argv[i];
        if (arg == "--probe")
        {
            args.probe = true;
        }
        else if (arg == "--engine-dir" && i + 1 < argc)
        {
            args.engineDir = argv[++i];
        }
        else if (arg == "--no-cuda-graph")
        {
            args.captureCudaGraph = false;
        }
        else if (arg == "--verbose")
        {
            args.verbose = true;
        }
        else
        {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (!args.probe && args.engineDir.empty())
    {
        throw std::invalid_argument("--engine-dir is required");
    }
    return args;
}

} // namespace

int main(int argc, char** argv)
{
    std::signal(SIGPIPE, SIG_IGN);
    // Keep the original stdout for protocol events and send everything else written to stdout
    // (Edge-LLM logs INFO there) to stderr, so logs can never corrupt the protocol stream.
    int const protocolFd = ::dup(STDOUT_FILENO);
    ::dup2(STDERR_FILENO, STDOUT_FILENO);
    EventWriter out(protocolFd);

    Args args;
    try
    {
        args = parseArgs(argc, argv);
    }
    catch (std::exception const& e)
    {
        logLine("error", e.what());
        return 64;
    }
    trt_edgellm::gLogger.setLevel(
        args.verbose ? nvinfer1::ILogger::Severity::kINFO : nvinfer1::ILogger::Severity::kWARNING);

    if (args.probe)
    {
        try
        {
            json event = deviceFacts();
            event["ev"] = "probe";
            out.push(event);
        }
        catch (std::exception const& e)
        {
            out.push({{"ev", "error"}, {"id", nullptr}, {"code", "no_device"}, {"message", e.what()}, {"fatal", true}});
            out.close();
            return 2;
        }
        out.close();
        return 0;
    }

    Worker worker(args.engineDir, args.captureCudaGraph, out);
    try
    {
        worker.load();
    }
    catch (std::exception const& e)
    {
        logLine("error", std::string("engine load failed: ") + e.what());
        out.push({{"ev", "error"}, {"id", nullptr}, {"code", "load_failed"}, {"message", e.what()}, {"fatal", true}});
        out.close();
        return 2;
    }

    // Generate commands run one at a time on this (main) thread; the reader thread answers
    // count/cancel immediately so admission checks and cancellation never wait on a generation.
    std::mutex queueMutex;
    std::condition_variable queueCv;
    std::deque<std::shared_ptr<Worker::Job>> jobs;
    bool stopping = false;

    std::thread reader([&] {
        LineReader input(STDIN_FILENO, kMaxCommandBytes);
        std::string line;
        bool tooLong = false;
        while (input.next(line, tooLong))
        {
            // Errors that cannot be attributed to a command carry no id, so they never end a job.
            if (tooLong)
            {
                out.push(errorEvent(nullptr, "bad_request",
                    "command longer than " + std::to_string(kMaxCommandBytes) + " bytes ignored", false));
                continue;
            }
            if (line.empty())
            {
                continue;
            }
            int64_t const received = nowNs();
            json cmd;
            try
            {
                cmd = json::parse(line);
            }
            catch (std::exception const& e)
            {
                out.push(errorEvent(nullptr, "bad_request", std::string("invalid JSON: ") + e.what(), false));
                continue;
            }
            if (!cmd.is_object())
            {
                out.push(errorEvent(nullptr, "bad_request", "a command must be a JSON object", false));
                continue;
            }
            auto const opField = cmd.find("op");
            std::string const op = opField != cmd.end() && opField->is_string() ? opField->get<std::string>() : "";
            if (op == "shutdown")
            {
                break;
            }
            json const id = cmd.value("id", json());
            try
            {
                if (op == "generate")
                {
                    auto job = worker.accept(std::move(cmd), received);
                    std::lock_guard<std::mutex> lock(queueMutex);
                    jobs.push_back(std::move(job));
                    queueCv.notify_one();
                }
                else if (op == "count")
                {
                    worker.count(cmd);
                }
                else if (op == "cancel")
                {
                    worker.cancel(cmd);
                }
                else
                {
                    out.push(errorEvent(id, "bad_request", "unknown op '" + op + "'", false));
                }
            }
            catch (std::exception const& e)
            {
                // A generate that could not be queued still needs its terminal event; a rejected
                // cancel must not end the job it names.
                out.push(op == "generate"
                        ? errorEvent(id, "generation_failed", e.what(), false)
                        : errorEvent(nullptr, "bad_request", "'" + op + "' command ignored: " + e.what(), false));
            }
        }
        // stdin closed or shutdown requested: stop the running job and drop queued ones. (A
        // draining front-end only sends shutdown once nothing is in flight.)
        worker.cancelAll();
        std::lock_guard<std::mutex> lock(queueMutex);
        stopping = true;
        queueCv.notify_one();
    });

    int exitCode = 0;
    while (true)
    {
        std::shared_ptr<Worker::Job> job;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            queueCv.wait(lock, [&] { return stopping || !jobs.empty(); });
            if (stopping)
            {
                break;
            }
            job = std::move(jobs.front());
            jobs.pop_front();
        }
        if (!worker.generate(*job))
        {
            exitCode = 3; // broken CUDA context: let the supervisor restart us
            break;
        }
    }

    out.close();
    if (exitCode != 0)
    {
        std::_Exit(exitCode); // the reader thread may still be blocked on stdin
    }
    reader.join();
    return 0;
}
