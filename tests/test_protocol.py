"""Request models, validation errors and response builders of the HTTP layer (no server needed)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from edgeserve.api.errors import from_validation
from edgeserve.api.protocol import (
    MAX_INT32,
    ChatCompletionRequest,
    CompletionRequest,
    chat_chunk,
    completion_chunk,
    resolve_sampling,
    valid_request_id,
)

USER = [{"role": "user", "content": "hi"}]


def chat_error(payload: dict):
    with pytest.raises(ValidationError) as info:
        ChatCompletionRequest.model_validate(payload)
    return from_validation(info.value)


def completion_error(payload: dict):
    with pytest.raises(ValidationError) as info:
        CompletionRequest.model_validate(payload)
    return from_validation(info.value)


# ---------------------------------------------------------------------------- request IDs

@pytest.mark.parametrize("value", ["a", "seg-000123", "a b", "x" * 512, "!~\"\\%/.."])
def test_valid_request_ids(value):
    assert valid_request_id(value)


@pytest.mark.parametrize("value", ["", "x" * 513, "tab\tid", "new\nline", "caf\u00e9", "nul\x00", "\x7f",
                                   " leading", "trailing ", " "])
def test_invalid_request_ids(value):
    # HTTP strips spaces around header values, so such an ID could not be echoed in X-Request-Id.
    assert not valid_request_id(value)


# ---------------------------------------------------------------------------- OpenAI nullable fields

def test_null_n_stream_and_echo_mean_the_defaults():
    request = ChatCompletionRequest.model_validate({"messages": USER, "n": None, "stream": None})
    assert request.stream is False
    completion = CompletionRequest.model_validate({"prompt": "x", "echo": None, "stream": None})
    assert completion.stream is False and not completion.echo
    assert chat_error({"messages": USER, "n": 2}).param == "n"
    assert completion_error({"prompt": "x", "echo": True}).param == "echo"


# ---------------------------------------------------------------------------- values the worker cannot take

@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens", "top_k"])
def test_integers_beyond_32_bits_are_refused(field):
    ChatCompletionRequest.model_validate({"messages": USER, field: MAX_INT32})
    for value in (MAX_INT32 + 1, int("9" * 4300)):  # the latter used to crash formatting an error message
        error = chat_error({"messages": USER, field: value})
        assert error.status == 400 and error.param == field and error.code == "invalid_value"


def test_unpaired_surrogates_are_refused_with_a_clear_message():
    # json.loads turns "\ud800" into a lone surrogate, which the worker's JSON parser rejects: it would never
    # answer the command (the request hung for 30 s and failed with a 500).
    for payload, param in (({"messages": USER, "stop": "\ud800"}, "stop"),
                           ({"messages": USER, "stop": ["ok", "x\udc00"]}, "stop"),
                           ({"messages": [{"role": "user", "content": "\ud800"}]}, "messages.0"),
                           ({"messages": [{"role": "user", "content": [{"type": "text", "text": "\udfff"}]}]},
                            "messages.0")):
        error = chat_error(payload)
        assert error.param == param and "unpaired surrogate" in error.message
        assert "codec" not in error.message
    error = completion_error({"prompt": "\ud800"})
    assert error.param == "prompt" and "unpaired surrogate" in error.message


def test_unpaired_surrogates_in_metadata_are_refused():
    # metadata is stored in the record, so it is returned by return_record and GET /v1/records: a lone surrogate
    # there made both fail with a 500 (UnicodeEncodeError while rendering the JSON response).
    for metadata in ({"task": "\ud83d"}, {"\udc00": "x"}):
        error = chat_error({"messages": USER, "metadata": metadata})
        assert error.param == "metadata" and "unpaired surrogate" in error.message
    ChatCompletionRequest.model_validate({"messages": USER, "metadata": {"task": "caf\u00e9 \U0001f600"}})


# ---------------------------------------------------------------------------- validation messages

def test_image_parts_are_reported_as_unsupported():
    error = chat_error({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:..."}}]}]})
    assert error.code == "unsupported_parameter" and error.param == "messages.0.content.1.image_url"


def test_union_member_names_do_not_leak_into_locations():
    cases = [
        ({"messages": [{"role": "user", "content": None}]}, "messages.0.content"),
        ({"messages": [{"role": "user", "content": 5}]}, "messages.0.content"),
        ({"messages": [{"role": "user", "content": ["plain"]}]}, "messages.0.content.0"),
        ({"messages": [{"role": "user", "content": [{"type": "text"}]}]}, "messages.0.content.0.text"),
        ({"messages": USER, "stop": 5}, "stop"),
        ({"messages": USER, "stop": ["a", 5]}, "stop.1"),
    ]
    for payload, param in cases:
        error = chat_error(payload)
        assert error.param == param, (payload, error.param)
        assert "str" not in error.param.split(".") and "[" not in error.message
    error = completion_error({"prompt": [1, 2, 3]})
    assert error.param == "prompt.0"
    assert completion_error({"prompt": None}).param == "prompt"


def test_messages_do_not_name_internal_classes():
    error = chat_error({"messages": ["hello"]})
    assert error.param == "messages.0" and "ChatMessage" not in error.message
    error = chat_error({"messages": [{"role": "user", "content": [7]}]})
    assert "TextPart" not in error.message


def test_unknown_keys_are_named_verbatim():
    for key in ("tools", "str", "foo-bar", "list[str]"):
        error = chat_error({"messages": USER, key: 1})
        assert error.code == "unsupported_parameter" and error.param == key
    error = chat_error({"messages": USER, "stream_options": {"include_usage": True, "x": 1}})
    assert error.param == "stream_options.x"


def test_error_bodies_are_always_encodable():
    for payload in ({"messages": USER, "metadata": {"\ud800": 1}}, {"messages": USER, "\udc80": 1}):
        error = chat_error(payload)
        json.dumps(error.body(), ensure_ascii=False).encode("utf-8")


# ---------------------------------------------------------------------------- sampling defaults

def test_sampling_defaults_tolerate_null_and_malformed_generation_config():
    sampled = ChatCompletionRequest.model_validate({"messages": USER, "temperature": 0.7})
    plain = ChatCompletionRequest.model_validate({"messages": USER})
    # People set temperature/top_p to null in generation_config.json to silence Hugging Face warnings.
    defaults = {"do_sample": True, "temperature": None, "top_p": None, "top_k": None}
    assert resolve_sampling(sampled, defaults) == {"temperature": 0.7, "top_p": 1.0, "top_k": 0}
    assert resolve_sampling(plain, defaults) == {"temperature": 0.0, "top_p": 1.0, "top_k": 1}
    garbage = {"temperature": "hot", "top_p": float("nan"), "top_k": 10**400}
    assert resolve_sampling(sampled, garbage) == {"temperature": 0.7, "top_p": 1.0, "top_k": 0}
    assert resolve_sampling(plain, {"temperature": float("inf")})["temperature"] == 0.0
    assert resolve_sampling(sampled, {"top_p": 0.9, "top_k": 50}) == {"temperature": 0.7, "top_p": 0.9, "top_k": 50}
    assert resolve_sampling(plain, {"temperature": 0.6, "top_p": 0.9}) == {"temperature": 0.6, "top_p": 0.9,
                                                                           "top_k": 0}


def test_a_greedy_generation_config_does_not_turn_requested_sampling_back_into_greedy():
    # The bundled models' generation_config.json: greedy, with top_k 1 (as they were evaluated).
    greedy = {"do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": 1, "max_new_tokens": 512}
    plain = ChatCompletionRequest.model_validate({"messages": USER})
    assert resolve_sampling(plain, greedy) == {"temperature": 0.0, "top_p": 1.0, "top_k": 1}
    # OpenAI clients have no top_k: temperature (and top_p) must take effect on their own.
    hot = ChatCompletionRequest.model_validate({"messages": USER, "temperature": 0.8})
    assert resolve_sampling(hot, greedy) == {"temperature": 0.8, "top_p": 1.0, "top_k": 0}
    nucleus = ChatCompletionRequest.model_validate({"messages": USER, "temperature": 0.8, "top_p": 0.9})
    assert resolve_sampling(nucleus, greedy) == {"temperature": 0.8, "top_p": 0.9, "top_k": 0}
    # A sampling generation config still provides its defaults.
    sampling = {"do_sample": True, "temperature": 0.6, "top_p": 0.9, "top_k": 40}
    assert resolve_sampling(hot, sampling) == {"temperature": 0.8, "top_p": 0.9, "top_k": 40}


@pytest.mark.parametrize("field", ["tools", "tool_choice", "response_format", "functions", "logit_bias",
                                   "top_logprobs", "parallel_tool_calls", "verbosity", "moderation",
                                   "prompt_cache_options", "prompt_cache_retention"])
def test_explicit_null_for_an_unsupported_feature_means_not_used(field):
    assert ChatCompletionRequest.model_validate({"messages": USER, field: None})  # some clients send every field
    error = chat_error({"messages": USER, field: [] if field != "parallel_tool_calls" else True})
    assert error.code == "unsupported_parameter" and error.param == field


def test_end_user_ids_and_cache_keys_are_accepted_and_ignored():
    for field in ("user", "safety_identifier", "prompt_cache_key"):
        assert ChatCompletionRequest.model_validate({"messages": USER, field: "u-123"})
        assert ChatCompletionRequest.model_validate({"messages": USER, field: None})


def test_explicit_null_for_an_unsupported_message_field_means_not_used():
    replayed = {"role": "assistant", "content": "ok", "tool_calls": None, "function_call": None, "refusal": None,
                "audio": None, "annotations": None}  # what model_dump() of an earlier response contains
    assert ChatCompletionRequest.model_validate({"messages": USER + [replayed] + USER})
    error = chat_error({"messages": USER + [{"role": "assistant", "content": "ok", "tool_calls": []}]})
    assert error.code == "unsupported_parameter" and error.param == "messages.1.tool_calls"


def test_stream_options_need_stream():
    error = chat_error({"messages": USER, "stream_options": {"include_usage": True}})
    assert error.code == "invalid_value" and "stream_options" in error.message
    assert ChatCompletionRequest.model_validate({"messages": USER, "stream": True,
                                                "stream_options": {"include_usage": True}}).include_usage
    assert ChatCompletionRequest.model_validate({"messages": USER, "stream_options": None})


# ---------------------------------------------------------------------------- stream chunks

def test_chunk_shapes_match_openai():
    role = chat_chunk("chatcmpl-x", "m", 1, "fp_1", {"role": "assistant", "content": ""})
    assert role["object"] == "chat.completion.chunk"
    assert role["choices"] == [{"index": 0, "delta": {"role": "assistant", "content": ""}, "logprobs": None,
                                "finish_reason": None}]
    last = chat_chunk("chatcmpl-x", "m", 1, "fp_1", {}, finish_reason="stop")
    assert last["choices"][0]["delta"] == {} and last["choices"][0]["finish_reason"] == "stop"
    usage = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    assert chat_chunk("chatcmpl-x", "m", 1, "fp_1", None, usage=usage)["choices"] == []
    text = completion_chunk("cmpl-x", "m", 1, "fp_1", "hi")
    assert text["object"] == "text_completion" and text["choices"][0]["text"] == "hi"
    final = completion_chunk("cmpl-x", "m", 1, "fp_1", None, usage=usage)
    assert final["choices"] == [] and final["usage"] == usage
