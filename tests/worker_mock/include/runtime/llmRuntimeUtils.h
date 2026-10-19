// The Edge-LLM 0.11 request / response / streaming types native/worker.cpp uses, with the same field
// names and defaults (temperature, topP, topK and maxGenerateLength have none upstream either).
#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

enum class FinishReason : uint8_t
{
    kNotFinished = 0,
    kEndId = 1,
    kLength = 2,
    kCancelled = 3,
    kError = 4,
    kStopWords = 5,
};

struct StreamChunk
{
    std::vector<int32_t> tokenIds;
    std::string text;
    bool finished{false};
    FinishReason reason{FinishReason::kNotFinished};
};

class StreamChannel
{
public:
    static std::shared_ptr<StreamChannel> create()
    {
        return std::shared_ptr<StreamChannel>(new StreamChannel());
    }

    std::optional<StreamChunk> tryPop();

    bool isFinished() const noexcept
    {
        return mFinished.load();
    }

    bool isCancelled() const noexcept
    {
        return mCancelled.load();
    }

    void cancel() noexcept
    {
        mCancelled.store(true);
    }

    void setStreamInterval(int32_t) noexcept {}

    // Producer side; upstream it is private and reached by the runtime through friendship.
    void push(StreamChunk chunk);
    void finish(FinishReason reason);
    //! Makes the empty tryPop that starts the consumer's next drain pass signal waitEmptyPop() and
    //! then stall, so the producer can publish in the window between that pop and the consumer's
    //! next look at its own state (MOCK:pump_race).
    void armEmptyPopHandshake();
    bool waitEmptyPop(int timeoutMs);

    std::atomic<bool> attached{false};

private:
    StreamChannel() = default;

    std::mutex mMutex;
    std::deque<StreamChunk> mPending;
    std::atomic<bool> mFinished{false};
    std::atomic<bool> mCancelled{false};
    bool mHandshakeArmed{false};
    int mEmptyPopsArmed{0};
    bool mEmptyPopSeen{false};
};

struct Message
{
    struct MessageContent
    {
        std::string type;
        std::string content;
    };

    std::string role;
    std::vector<MessageContent> contents;
};

struct TokenCallbackInfo
{
    int32_t tokenId;
    int32_t batchIdx;
    int32_t generationStep;
    bool isFinished;
};

using TokenCallback = std::function<void(TokenCallbackInfo const&)>;

struct LLMGenerationRequest
{
    struct Request
    {
        std::vector<Message> messages;
        std::vector<std::string> stopStrings;
        std::optional<uint64_t> samplingSeed;
    };

    std::vector<Request> requests;
    std::vector<std::vector<int32_t>> preTokenizedInputIds;
    float temperature;
    float topP;
    int64_t topK;
    int64_t maxGenerateLength;
    std::optional<uint64_t> samplingSeed;
    bool applyChatTemplate{true};
    bool addGenerationPrompt{true};
    bool skipSpecialTokens{true};
    std::vector<std::shared_ptr<StreamChannel>> streamChannels;
    std::optional<TokenCallback> onTokenGenerated;
};

struct LLMGenerationResponse
{
    std::vector<std::vector<int32_t>> outputIds;
    std::vector<std::string> outputTexts;
    std::vector<FinishReason> finishReasons;
    std::vector<int32_t> inputTokenCounts;
};

} // namespace rt
} // namespace trt_edgellm
