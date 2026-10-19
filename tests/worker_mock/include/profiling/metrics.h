#pragma once

#include <string>

namespace trt_edgellm
{
namespace metrics
{
namespace StageNames
{
inline std::string const kLLM_PREFILL = "llm_prefill";
inline std::string const kLLM_GENERATION = "llm_generation";
} // namespace StageNames
} // namespace metrics
} // namespace trt_edgellm
