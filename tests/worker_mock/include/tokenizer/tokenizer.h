// Whitespace tokenizer: one id per distinct word, in a table shared by all threads (count runs on
// the worker's reader thread while a generation runs on its main thread).
#pragma once

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace tokenizer
{
using Rank = std::int32_t;

inline constexpr Rank kMockBos = 1;
inline constexpr Rank kMockEos = 2;

class Tokenizer
{
public:
    bool loadFromHF(std::filesystem::path const& modelDir)
    {
        return std::filesystem::exists(modelDir / "tokenizer.json");
    }

    std::vector<Rank> encode(std::string const& text, bool addBos = false, bool = false) const
    {
        std::vector<Rank> ids;
        if (addBos)
        {
            ids.push_back(kMockBos);
        }
        std::istringstream words(text);
        std::string word;
        while (words >> word)
        {
            ids.push_back(word == "<|begin_of_text|>" ? kMockBos : idOf(word));
        }
        return ids;
    }

    std::string idToPiece(Rank token, bool skipSpecialTokens = true) const
    {
        if (token == kMockBos || token == kMockEos)
        {
            return skipSpecialTokens ? "" : (token == kMockBos ? "<|begin_of_text|>" : "<|eot_id|>");
        }
        std::lock_guard<std::mutex> lock(table().mutex);
        auto const index = static_cast<size_t>(token - 100);
        return index < table().words.size() ? table().words[index] : "<unk>";
    }

    Rank getBosId() const noexcept
    {
        return kMockBos;
    }

    static Rank idOf(std::string const& word)
    {
        std::lock_guard<std::mutex> lock(table().mutex);
        auto const it = table().ids.find(word);
        if (it != table().ids.end())
        {
            return it->second;
        }
        Rank const id = static_cast<Rank>(100 + table().words.size());
        table().words.push_back(word);
        table().ids.emplace(word, id);
        return id;
    }

private:
    struct Table
    {
        std::mutex mutex;
        std::unordered_map<std::string, Rank> ids;
        std::vector<std::string> words;
    };

    static Table& table()
    {
        static Table instance;
        return instance;
    }
};
} // namespace tokenizer
} // namespace trt_edgellm
