// Deterministic stand-in for Edge-LLM's LLMInferenceRuntime, in Edge-LLM 0.11's order of events: the
// prefill samples one token per sequence and fires its token callbacks, cancellation is observed
// right after the prefill and at the top of every decode step (so a sequence cancelled before it
// started still produces one token), then finish states are updated, stream chunks pushed and,
// for decode steps, the token callbacks fired; finished sequences leave the batch, which compacts.
//
// Words in a prompt steer it:
//   MOCK:tokens=N        N tokens, the last one end-of-sequence ($MOCK_TOKENS, else 6); with
//                        max_new_tokens <= N it ends at max_new_tokens with "length" instead
//   MOCK:step_ms=N       duration of each step ($MOCK_STEP_MS, else 2)
//   MOCK:fail            fail (return false) after the prefill
//   MOCK:fail_nonsticky  the same, leaving a non-sticky CUDA error (out of memory) as the thread's last error
//   MOCK:sticky          the same with a sticky CUDA error: the context stays unusable
//   MOCK:throw, MOCK:throw_unknown  throw a std::runtime_error / a non-std exception
//   MOCK:too_long        throw Edge-LLM's EDGELLM_INPUT_TOO_LONG error
//   MOCK:pump_race       publish a sequence's last chunk only once the consumer, at the start of a
//                        drain pass, found the channel empty, and while it is still inside tryPop
//   MOCK:utf8            every token is "\u00e9" and each stream chunk holds back the last byte, cutting the
//                        character in two (Edge-LLM's stop-string look-ahead), never flushed at the end
//   MOCK:utf8_cut        the token at max_new_tokens is half of a 4-byte character (cut by the length limit)
//   MOCK:echo_sampling   the output text is the sampling parameters the runtime received
#pragma once

#include "profiling/metrics.h"
#include "profiling/timer.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/streaming.h"
#include "tokenizer/tokenizer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

class LLMInferenceRuntime
{
public:
    //! $MOCK_MAX_BATCH sets the engine's batch size (default 4).
    LLMInferenceRuntime(std::string const& engineDir, std::string const&,
        std::unordered_map<std::string, std::string> const&, cudaStream_t)
    {
        if (!std::filesystem::exists(std::filesystem::path(engineDir) / "llm.engine"))
        {
            throw std::runtime_error("no llm.engine in " + engineDir);
        }
        mMaxBatch = static_cast<int32_t>(envLong("MOCK_MAX_BATCH", 4));
    }

    bool captureDecodingCUDAGraph(cudaStream_t)
    {
        return true;
    }

    int32_t maxBatchSize() const
    {
        return mMaxBatch;
    }

    std::vector<int32_t> countPromptTokens(LLMGenerationRequest const& request) const
    {
        std::vector<int32_t> counts;
        for (auto const& item : request.requests)
        {
            counts.push_back(static_cast<int32_t>(templated(item).size()));
        }
        return counts;
    }

    bool handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t)
    {
        response = LLMGenerationResponse{};
        // Edge-LLM checks cudaGetLastError() after its first kernel launch; runInline turns the throw into false.
        if (cudaGetLastError() != cudaSuccess)
        {
            std::cerr << "[mock-runtime] CUDA error at the first launch check" << std::endl;
            return false;
        }
        size_t const n = request.requests.size();
        if (n == 0 || static_cast<int32_t>(n) > mMaxBatch)
        {
            return false;
        }
        if (!request.streamChannels.empty())
        {
            if (request.streamChannels.size() != n)
            {
                return false;
            }
            for (auto const& channel : request.streamChannels)
            {
                if (channel && (channel->isFinished() || channel->attached.load()))
                {
                    return false;
                }
            }
            for (auto const& channel : request.streamChannels)
            {
                if (channel)
                {
                    channel->attached = true;
                }
            }
        }
        int32_t const maxGenerateLength = request.maxGenerateLength; // narrowed as in Edge-LLM
        int32_t const limit = std::max(1, maxGenerateLength);

        std::vector<Slot> slots(n);
        std::string mode;
        tokenizer::Tokenizer const tok;
        for (size_t i = 0; i < n; ++i)
        {
            Slot& s = slots[i];
            s.prompt = i < request.preTokenizedInputIds.size() && !request.preTokenizedInputIds[i].empty()
                ? request.preTokenizedInputIds[i]
                : templated(request.requests[i]);
            s.want = envLong("MOCK_TOKENS", 6);
            s.stepMs = envLong("MOCK_STEP_MS", 2);
            if (!request.streamChannels.empty())
            {
                s.channel = request.streamChannels[i];
            }
            for (int32_t id : s.prompt)
            {
                std::string const word = tok.idToPiece(id, false);
                if (word.rfind("MOCK:tokens=", 0) == 0)
                {
                    s.want = std::atol(word.c_str() + 12);
                }
                else if (word.rfind("MOCK:step_ms=", 0) == 0)
                {
                    s.stepMs = std::atol(word.c_str() + 13);
                }
                else if (word == "MOCK:pump_race")
                {
                    s.race = true;
                }
                else if (word == "MOCK:utf8")
                {
                    s.utf8 = true;
                }
                else if (word == "MOCK:utf8_cut")
                {
                    s.utf8Cut = true;
                }
                else if (word.rfind("MOCK:", 0) == 0)
                {
                    mode = word.substr(5);
                }
            }
        }
        if (mode == "too_long")
        {
            throw std::runtime_error("EDGELLM_INPUT_TOO_LONG: input length exceeds engine max_kv_cache_capacity");
        }
        if (mode == "throw")
        {
            throw std::runtime_error("mock runtime failure");
        }
        if (mode == "throw_unknown")
        {
            throw 42;
        }

        std::vector<size_t> active;
        for (size_t i = 0; i < n; ++i)
        {
            active.push_back(i);
        }
        std::vector<std::string> deltas(n);

        // Prefill.
        sleepMs(slots[0].stepMs);
        gTimer.record(metrics::StageNames::kLLM_PREFILL, 0.9F * static_cast<float>(slots[0].stepMs) + 0.25F);
        for (size_t i = 0; i < active.size(); ++i)
        {
            Slot& s = slots[active[i]];
            deltas[active[i]] = append(s, limit);
            callback(request, s.ids.back(), static_cast<int32_t>(i), 0, false);
        }
        if (mode == "fail" || mode == "fail_nonsticky" || mode == "sticky")
        {
            if (mode == "fail_nonsticky")
            {
                mockcuda::lastError = cudaErrorMemoryAllocation;
            }
            if (mode == "sticky")
            {
                mockcuda::stickyError = cudaErrorIllegalAddress;
                mockcuda::lastError = cudaErrorIllegalAddress;
            }
            finalize(slots);
            return false;
        }
        applyCancellation(slots, active);
        updateFinish(slots, active, limit);
        for (size_t index : active)
        {
            emit(slots[index], deltas[index]);
        }
        evict(slots, active);

        // Decode steps.
        for (int32_t round = 1; !active.empty(); ++round)
        {
            applyCancellation(slots, active);
            sleepMs(slots[active[0]].stepMs);
            gTimer.record(metrics::StageNames::kLLM_GENERATION, 0.9F * static_cast<float>(slots[active[0]].stepMs));
            std::vector<bool> produced(active.size(), false);
            for (size_t i = 0; i < active.size(); ++i)
            {
                Slot& s = slots[active[i]];
                deltas[active[i]].clear();
                if (s.reason == FinishReason::kNotFinished)
                {
                    deltas[active[i]] = append(s, limit);
                    produced[i] = true;
                }
            }
            updateFinish(slots, active, limit);
            for (size_t index : active)
            {
                emit(slots[index], deltas[index]);
            }
            for (size_t i = 0; i < active.size(); ++i)
            {
                if (produced[i])
                {
                    Slot const& s = slots[active[i]];
                    callback(request, s.ids.back(), static_cast<int32_t>(i), round,
                        s.reason != FinishReason::kNotFinished);
                }
            }
            evict(slots, active);
        }
        finalize(slots);

        for (Slot const& s : slots)
        {
            response.outputIds.push_back(s.ids);
            if (mode == "echo_sampling")
            {
                char buffer[128];
                std::snprintf(buffer, sizeof(buffer), "temperature=%.3f top_k=%lld top_p=%.9f", request.temperature,
                    static_cast<long long>(request.topK), request.topP);
                response.outputTexts.push_back(buffer);
            }
            else
            {
                response.outputTexts.push_back(s.text);
            }
            response.finishReasons.push_back(s.reason);
            response.inputTokenCounts.push_back(static_cast<int32_t>(s.prompt.size()));
        }
        return true;
    }

private:
    struct Slot
    {
        std::vector<int32_t> prompt;
        long want{6};
        long stepMs{2};
        bool race{false};
        bool utf8{false};
        bool utf8Cut{false};
        std::string held; // the byte a MOCK:utf8 stream holds back
        std::vector<int32_t> ids;
        std::string text;
        FinishReason reason{FinishReason::kNotFinished};
        std::shared_ptr<StreamChannel> channel;
    };

    //! Samples the next token: "w<k>", or end-of-sequence as token MOCK:tokens if that comes first.
    static std::string append(Slot& s, int32_t limit)
    {
        long const k = static_cast<long>(s.ids.size());
        if (s.want < limit && k == s.want - 1)
        {
            s.ids.push_back(tokenizer::kMockEos);
            return "";
        }
        if (s.utf8)
        {
            s.ids.push_back(tokenizer::Tokenizer::idOf("e"));
            s.text += "\xC3\xA9";
            std::string const out = s.held + "\xC3\xA9";
            s.held = out.substr(out.size() - 1);
            return out.substr(0, out.size() - 1);
        }
        if (s.utf8Cut && k == limit - 1)
        {
            s.ids.push_back(tokenizer::Tokenizer::idOf("rocket"));
            s.text += "\xF0\x9F";
            return "\xF0\x9F";
        }
        std::string const word = "w" + std::to_string(k);
        s.ids.push_back(tokenizer::Tokenizer::idOf(word));
        s.text += word + " ";
        return word + " ";
    }

    static void applyCancellation(std::vector<Slot>& slots, std::vector<size_t> const& active)
    {
        for (size_t index : active)
        {
            Slot& s = slots[index];
            if (s.reason == FinishReason::kNotFinished && s.channel && s.channel->isCancelled())
            {
                s.reason = FinishReason::kCancelled;
            }
        }
    }

    static void updateFinish(std::vector<Slot>& slots, std::vector<size_t> const& active, int32_t limit)
    {
        for (size_t index : active)
        {
            Slot& s = slots[index];
            if (s.reason != FinishReason::kNotFinished)
            {
                continue;
            }
            if (s.ids.back() == tokenizer::kMockEos)
            {
                s.reason = FinishReason::kEndId;
            }
            else if (static_cast<int32_t>(s.ids.size()) >= limit)
            {
                s.reason = FinishReason::kLength;
            }
        }
    }

    static void emit(Slot& s, std::string const& delta)
    {
        if (!s.channel)
        {
            return;
        }
        bool const finished = s.reason != FinishReason::kNotFinished;
        if (finished && s.race)
        {
            s.channel->armEmptyPopHandshake();
            s.channel->waitEmptyPop(3000);
        }
        s.channel->push({{}, delta, finished, s.reason});
        if (finished)
        {
            s.channel->finish(s.reason);
        }
    }

    static void evict(std::vector<Slot> const& slots, std::vector<size_t>& active)
    {
        std::vector<size_t> still;
        for (size_t index : active)
        {
            if (slots[index].reason == FinishReason::kNotFinished)
            {
                still.push_back(index);
            }
        }
        active.swap(still);
    }

    //! Like Edge-LLM's StreamChannelFinalizer: every channel ends, with kError if nothing ended it.
    static void finalize(std::vector<Slot>& slots)
    {
        for (Slot& s : slots)
        {
            if (s.channel && !s.channel->isFinished())
            {
                s.channel->push({{}, "", true, FinishReason::kError});
                s.channel->finish(FinishReason::kError);
            }
        }
    }

    static void callback(
        LLMGenerationRequest const& request, int32_t token, int32_t batchIdx, int32_t step, bool finished)
    {
        if (request.onTokenGenerated.has_value())
        {
            (*request.onTokenGenerated)(TokenCallbackInfo{token, batchIdx, step, finished});
        }
    }

    static std::vector<int32_t> templated(LLMGenerationRequest::Request const& item)
    {
        tokenizer::Tokenizer const tok;
        std::vector<int32_t> ids{tok.getBosId()};
        for (auto const& message : item.messages)
        {
            ids.push_back(tokenizer::Tokenizer::idOf("<|start_header_id|>" + message.role + "<|end_header_id|>"));
            for (auto const& content : message.contents)
            {
                auto const words = tok.encode(content.content);
                ids.insert(ids.end(), words.begin(), words.end());
            }
            ids.push_back(tokenizer::Tokenizer::idOf("<|eot_id|>"));
        }
        ids.push_back(tokenizer::Tokenizer::idOf("<|start_header_id|>assistant<|end_header_id|>"));
        return ids;
    }

    static long envLong(char const* name, long fallback)
    {
        char const* value = std::getenv(name);
        return value != nullptr ? std::atol(value) : fallback;
    }

    static void sleepMs(long ms)
    {
        if (ms > 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));
        }
    }

    int32_t mMaxBatch{4};
};

} // namespace rt
} // namespace trt_edgellm
