"""Model folders: validation, architecture facts, cost model (FLOPs / bytes) and memory estimates.

A model folder is a TensorRT Edge-LLM ONNX export directory: the ``llm`` folder that
``tensorrt-edgellm-export <checkpoint> <output>`` writes into ``<output>``, e.g.::

    config.json                 Edge-LLM config (edgellm_version, model, layer counts, eos ids, ...)
    model.onnx (+ model.onnx.data)
    embedding.safetensors       fp16 [vocab, hidden] input embedding table (looked up by the runtime)
    tokenizer.json, tokenizer_config.json
    chat_template.jinja         the model's chat template (Edge-LLM needs it to build and run the engine)
    <external weight files>     listed in config["external_weight_files"] (e.g. INT4 FFN weights)
    vocab_map.safetensors       required when config["reduced_vocab_size"] is set
    generation_config.json      optional sampling defaults (Hugging Face format)

The cost model counts FLOPs as 2 per multiply-accumulate and reads every linear layer's
shape from the graph itself (INT4 plugin attributes gemm_n/gemm_k and Gemm/MatMul weight
initializers), never from config.json, so per-layer pruned widths and swapped models are
handled automatically. It covers dense transformer decoders; for other architectures (MoE,
hybrid SSM/linear attention) the FLOPs/bytes methods return None instead of a wrong number.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import operator
import os
import re
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import onnx


class ModelError(ValueError):
    """The folder is not a servable model; the message says what is wrong and how to fix it."""


_DENSE = "dense_transformer"
_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ModelInfo:
    """Facts about a validated model folder, as returned by :func:`load_model`."""

    name: str  # folder name; served as the OpenAI "model" id
    path: Path
    model_type: str  # config["model"], e.g. "llama"
    edgellm_version: str | None  # config["edgellm_version"]
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int  # input vocabulary (embedding rows)
    output_vocab_size: int  # lm_head rows (reduced_vocab_size when present)
    kv_cache_dtype: str  # config["kv_cache_dtype"], e.g. "fp16"
    kv_bytes_per_token: int  # 2 (K,V) x layers x kv_heads x head_dim x dtype bytes
    eos_token_ids: tuple[int, ...]
    quantization: str  # human-readable, e.g. "INT4 groupwise W4A16 (group 128), fp16 lm_head"
    linear_params: int  # all matmul weight elements incl. lm_head (excludes the embedding table)
    lm_head_params: int
    embedding_params: int
    resident_weight_bytes: int  # bytes the runtime keeps on the device: engine weights + external + embedding + vocab map
    streamed_weight_bytes: int  # weight bytes read per forward pass (everything except the embedding table)
    file_bytes: dict[str, int]  # size of every file in the folder
    has_chat_template: bool
    generation_defaults: dict  # parsed generation_config.json ({} when absent)
    cost_model: str  # "dense_transformer" | "unavailable"
    cost_model_reason: str | None  # why the cost model is unavailable
    mlp_widths: tuple[int, ...] | None = None  # each layer's MLP width, read from the graph (differs when pruned)

    def decode_flops(self, context_len: int) -> int | None:
        """FLOPs of one decode pass producing one token whose attention covers ``context_len`` keys.

        = 2 * linear_params + 4 * num_layers * num_heads * head_dim * context_len
        (QK^T and AV are 2 * num_heads * head_dim * context_len each per layer; GQA does not reduce them.)
        """
        context_len = _count("context_len", context_len)
        if self.cost_model != _DENSE:
            return None
        return 2 * self.linear_params + self._attention_flops_per_pair() * context_len

    def prefill_flops(self, new_tokens: int, cached_tokens: int = 0) -> int | None:
        """FLOPs of a prefill over ``new_tokens`` after ``cached_tokens`` reused ones (logits for the last token only).

        = 2 * (linear_params - lm_head_params) * new_tokens + 2 * lm_head_params
          + 4 * num_layers * num_heads * head_dim * (new_tokens * cached_tokens + new_tokens * (new_tokens + 1) / 2)
        """
        new_tokens = _count("new_tokens", new_tokens)
        cached_tokens = _count("cached_tokens", cached_tokens)
        if self.cost_model != _DENSE:
            return None
        attended_pairs = new_tokens * cached_tokens + new_tokens * (new_tokens + 1) // 2
        return (
            2 * (self.linear_params - self.lm_head_params) * new_tokens
            + 2 * self.lm_head_params
            + self._attention_flops_per_pair() * attended_pairs
        )

    def decode_bytes(self, context_len: int) -> int | None:
        """DRAM bytes moved by one decode pass: streamed weights + KV for ``context_len`` positions
        (context_len - 1 read, 1 written) + fp32 logits (4 * output_vocab_size)."""
        context_len = _count("context_len", context_len)
        if self.cost_model != _DENSE:
            return None
        return self.streamed_weight_bytes + self.kv_bytes_per_token * context_len + 4 * self.output_vocab_size

    def prefill_bytes(self, new_tokens: int, cached_tokens: int = 0) -> int | None:
        """DRAM bytes moved by a prefill: streamed weights + KV read for cached_tokens + KV written for new_tokens
        + fp32 logits for one row. Activations are not counted."""
        new_tokens = _count("new_tokens", new_tokens)
        cached_tokens = _count("cached_tokens", cached_tokens)
        if self.cost_model != _DENSE:
            return None
        return (
            self.streamed_weight_bytes
            + self.kv_bytes_per_token * (cached_tokens + new_tokens)
            + 4 * self.output_vocab_size
        )

    def as_dict(self) -> dict:
        """JSON-serialisable summary for records, /v1/models and run summaries (path as str, no file_bytes)."""
        out: dict[str, Any] = {}
        for item in fields(self):
            if item.name == "file_bytes":
                continue
            value = getattr(self, item.name)
            if item.name == "path":
                value = str(value)
            elif item.name in ("eos_token_ids", "mlp_widths"):
                value = None if value is None else list(value)
            elif item.name == "generation_defaults":
                value = copy.deepcopy(value)
            out[item.name] = value
        return out

    def _attention_flops_per_pair(self) -> int:
        """FLOPs of QK^T plus AV for one (query, key) pair summed over layers."""
        return 4 * self.num_layers * self.num_heads * self.head_dim


def _count(name: str, value: int) -> int:
    """``value`` as a non-negative int (TypeError for non-integers, ValueError when negative)."""
    try:
        number = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from None
    if number < 0:
        raise ValueError(f"{name} must be >= 0, got {number}")
    return number


# --------------------------------------------------------------------------- folder validation

_REQUIRED_FILES: dict[str, str] = {
    "model.onnx": "the ONNX graph",
    "embedding.safetensors": "the input embedding table",
    "tokenizer.json": "the tokenizer",
    "tokenizer_config.json": "the tokenizer settings",
}
_EMBEDDING = "embedding.safetensors"
_VOCAB_MAP = "vocab_map.safetensors"
_CHAT_TEMPLATE = "chat_template.jinja"
_GENERATION_CONFIG = "generation_config.json"
_EXPORT_HINT = "the 'llm' folder written by tensorrt-edgellm-export"
_REEXPORT_HINT = "re-export the model with tensorrt-edgellm-export"

# Bytes per element for the kv_cache_dtype values the Edge-LLM runtime accepts.
_KV_DTYPE_BYTES: dict[str, int] = {"fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "fp32": 4}


@dataclass(frozen=True)
class _ExternalWeights:
    file: str  # path relative to the model folder (POSIX)
    tensors: tuple[str, ...]  # graph inputs whose data the runtime loads from this file


def _reject_constant(name: str) -> None:
    raise ValueError(f"{name} is not allowed in JSON")


def _read_json_object(path: Path, what: str) -> dict:
    """Strict JSON: NaN / Infinity are rejected, as the runtime's parser does (and as_dict must stay valid JSON)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=_reject_constant)
    except OSError as exc:
        raise ModelError(f"cannot read {path} ({exc})") from exc
    except ValueError as exc:
        raise ModelError(f"{path} is not valid JSON ({exc}); {what}") from exc
    if not isinstance(data, dict):
        raise ModelError(f"{path} must contain a JSON object, found {type(data).__name__}; {what}")
    return data


def _read_config(root: Path) -> dict:
    if not root.exists():
        raise ModelError(f"model folder {root} does not exist")
    if not root.is_dir():
        hint = f" (did you mean its folder {root.parent}?)" if root.suffix in (".json", ".onnx") else ""
        raise ModelError(f"{root} is not a directory{hint}; pass {_EXPORT_HINT}")
    config_path = root / "config.json"
    if not config_path.is_file():
        nested = root / "llm"
        hint = f" ({nested} has one: did you mean that folder?)" if (nested / "config.json").is_file() else ""
        raise ModelError(
            f"{root} has no config.json, so it is not a TensorRT Edge-LLM export{hint}; pass {_EXPORT_HINT} "
            "(it contains config.json and model.onnx)"
        )
    return _read_json_object(config_path, "copy it again from the export")


def _check_engine_kind(root: Path, config: dict) -> None:
    spec = config.get("spec_decode_type")
    if spec is not None and spec != "none":
        raise ModelError(
            f"{root} was exported for speculative decoding (config.json spec_decode_type {spec!r}), which edgeserve "
            "does not support; export the model again without speculative decoding"
        )
    role = config.get("engine_role")
    if role is not None and role != "llm":
        raise ModelError(
            f"{root} is an Edge-LLM {role!r} component (config.json engine_role), not a plain language model; "
            f"pass {_EXPORT_HINT}"
        )


def _check_required_files(root: Path) -> None:
    missing = [f"{name} ({what})" for name, what in _REQUIRED_FILES.items() if not (root / name).is_file()]
    if missing:
        raise ModelError(
            f"{root} is missing {', '.join(missing)}; copy the complete folder written by tensorrt-edgellm-export "
            "(tokenizer files can also be copied from the original Hugging Face model)"
        )


def _safe_relpath(name: str) -> Path | None:
    """``name`` as a path inside the model folder, or None if it is empty, absolute or climbs out with ``..``."""
    if not name or "\0" in name:
        return None
    rel = Path(name)
    if rel.is_absolute() or ".." in rel.parts or rel == Path("."):
        return None
    return rel


def _external_weight_files(root: Path, config: dict) -> list[_ExternalWeights]:
    raw = config.get("external_weight_files")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ModelError('config.json: external_weight_files must be a list of {"file": ...} entries')
    entries = []
    for index, item in enumerate(raw):
        # The Edge-LLM builder only accepts {"file": "<name>", ...} objects; anything else fails the engine build.
        name = item.get("file") if isinstance(item, dict) else None
        rel = _safe_relpath(name) if isinstance(name, str) else None
        if rel is None:
            raise ModelError(
                f'config.json: external_weight_files[{index}] must be an object whose "file" names a file inside '
                f"the model folder, found {item!r:.80}; re-export the model"
            )
        tensors = item.get("tensors")
        names = tuple(t for t in tensors if isinstance(t, str)) if isinstance(tensors, list) else ()
        entries.append(_ExternalWeights(rel.as_posix(), names))
    missing = sorted({entry.file for entry in entries if not (root / entry.file).is_file()})
    if missing:
        raise ModelError(
            f"{root} is missing {', '.join(missing)}, listed in config.json external_weight_files; copy it from the "
            "export folder (tensorrt-edgellm-export --externalize-weights writes it next to model.onnx)"
        )
    return entries


def _reduced_vocab_size(root: Path, config: dict) -> int | None:
    value = config.get("reduced_vocab_size")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelError(f"config.json: reduced_vocab_size must be an integer, found {value!r}")
    if value <= 0:  # the runtime treats 0 as "full vocabulary"
        return None
    if not (root / _VOCAB_MAP).is_file():
        raise ModelError(
            f"{root} sets reduced_vocab_size={value} in config.json but {_VOCAB_MAP} is missing; copy it from the "
            "export (tensorrt-edgellm-reduce-vocab writes it) or export the model with its full vocabulary"
        )
    return value


def _read_generation_config(root: Path) -> dict:
    path = root / _GENERATION_CONFIG
    if not path.is_file():
        return {}
    return _read_json_object(path, "fix or delete it (it is optional)")


def _config_int(config: dict, key: str) -> int:
    """A required positive integer (the Edge-LLM builder or runtime reads it without a default)."""
    value = config.get(key)
    if value is None:
        raise ModelError(f"config.json has no {key}, which TensorRT Edge-LLM requires; {_REEXPORT_HINT}")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelError(f"config.json: {key} must be a positive integer, found {value!r}; {_REEXPORT_HINT}")
    return value


def _kv_cache_dtype(config: dict) -> tuple[str, int]:
    """(dtype, bytes per element); required, and matched exactly like the runtime's parser (lower case only)."""
    value = config.get("kv_cache_dtype")
    if value is None:
        raise ModelError(
            f"config.json has no kv_cache_dtype, which the TensorRT Edge-LLM runtime requires; {_REEXPORT_HINT}"
        )
    if not isinstance(value, str) or value not in _KV_DTYPE_BYTES:
        raise ModelError(
            f"config.json: kv_cache_dtype {value!r} is not supported (expected one of {', '.join(_KV_DTYPE_BYTES)})"
        )
    return value, _KV_DTYPE_BYTES[value]


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _kv_bytes_per_token(config: dict, num_layers: int, num_kv_heads: int, head_dim: int, dtype_bytes: int) -> int:
    """KV bytes per token as the Edge-LLM builder sizes the cache.

    With a ``kv_layer_configs`` table: one KV layer per object entry (null = recurrent layer without KV), whose
    num_kv_heads / head_dim default to the global ones. Without it: ``num_attention_layers`` layers for hybrid
    exports (num_linear_attn_layers > 0), every layer otherwise.
    """
    table = config.get("kv_layer_configs")
    if isinstance(table, list) and table:
        total = 0
        for layer in table:
            if layer is None:
                continue
            if not isinstance(layer, dict):
                break
            heads, dim = layer.get("num_kv_heads", num_kv_heads), layer.get("head_dim", head_dim)
            if not (_is_positive_int(heads) and _is_positive_int(dim)):
                break
            total += 2 * heads * dim * dtype_bytes
        else:
            return total
    layers = num_layers
    attention_layers = config.get("num_attention_layers")
    if _is_positive_int(config.get("num_linear_attn_layers")) and _is_count(attention_layers):
        layers = attention_layers
    return 2 * layers * num_kv_heads * head_dim * dtype_bytes


def _token_ids(value: object) -> list[int] | None:
    """An int or list of ints as a list ([] for None); None when malformed."""
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return list(values)
    return None


def _eos_token_ids(config: dict, generation: dict) -> tuple[int, ...]:
    ids = _token_ids(config.get("eos_token_id"))
    if ids is None:
        raise ModelError(
            f"config.json: eos_token_id must be an integer or a list of integers, found {config['eos_token_id']!r}"
        )
    extra = _token_ids(generation.get("eos_token_id")) or []  # optional file: malformed ids are ignored
    return tuple(dict.fromkeys(ids + extra))


# --------------------------------------------------------------------------- ONNX graph

def _load_onnx(path: Path) -> onnx.ModelProto:
    import onnx

    try:
        model = onnx.load(str(path), load_external_data=False)
    except Exception as exc:  # protobuf DecodeError, OSError, ... from a third-party parser
        raise ModelError(
            f"{path} is not a readable ONNX model ({type(exc).__name__}: {exc}); copy it again or re-export the model"
        ) from exc
    if not model.graph.node:
        raise ModelError(f"{path} contains no graph nodes; copy it again or re-export the model")
    return model


def _graph_tensors(graph: onnx.GraphProto) -> Iterator[onnx.TensorProto]:
    """Every tensor stored in a graph: initializers, sparse initializers, attribute tensors and subgraphs."""
    yield from graph.initializer
    for sparse in graph.sparse_initializer:
        yield sparse.values
        yield sparse.indices
    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("t"):
                yield attr.t
            yield from attr.tensors
            if attr.HasField("g"):
                yield from _graph_tensors(attr.g)
            for subgraph in attr.graphs:
                yield from _graph_tensors(subgraph)


def _check_external_data(root: Path, model: onnx.ModelProto) -> tuple[list[Path], int]:
    """(the external data files model.onnx references, the bytes of them it uses); ModelError if one is missing,
    truncated or outside root. The bytes in use can be fewer than the files hold: exporters that move weights
    out of model.onnx (--externalize-weights) leave their old bytes in the file."""
    from onnx import TensorProto

    needed: dict[str, int] = {}  # file -> minimum size
    referenced = 0
    for tensor in _graph_tensors(model.graph):
        if tensor.data_location != TensorProto.EXTERNAL:
            continue
        info = {entry.key: entry.value for entry in tensor.external_data}
        location = info.get("location", "")
        rel = _safe_relpath(location)
        if rel is None:
            raise ModelError(
                f"model.onnx tensor {tensor.name!r} keeps its data in {location!r}, which is not a file inside the "
                "model folder; re-export the model"
            )
        try:
            length = int(info.get("length") or 0)
            end = int(info.get("offset") or 0) + length
        except ValueError:
            length = end = 0
        referenced += length
        key = rel.as_posix()
        needed[key] = max(needed.get(key, 0), end)
    files = []
    for name in sorted(needed):
        path = root / name
        if not path.is_file():
            raise ModelError(
                f"model.onnx keeps its weights in {name}, which is missing from {root}; copy it next to model.onnx"
            )
        size = path.stat().st_size
        if size < needed[name]:
            raise ModelError(
                f"{path} is truncated ({size} bytes, model.onnx needs at least {needed[name]}); copy it again"
            )
        files.append(path)
    return files, referenced


@dataclass(frozen=True)
class _Linear:
    index: int  # position in graph.node
    params: int
    out_dim: int | None  # output features, None when the weight's layout cannot be read
    fmt: str  # weight format key (see _format_label)
    group_size: int | None = None
    zero_point_correction: bool = False  # the FP16 zero-point part of an asymmetric INT4 layer (edgeserve export)


_DEQUANTIZE_OPS = frozenset({"DequantizeLinear", "TRT_MXFP8DequantizeLinear"})
_RESHAPE_OPS = frozenset({"Reshape", "Flatten", "Squeeze", "Unsqueeze"})
# Ops that may sit between a weight tensor and the Gemm/MatMul consuming it (followed through input 0).
_WEIGHT_PATH_OPS = _DEQUANTIZE_OPS | _RESHAPE_OPS | {"Cast", "CastLike", "Identity", "Transpose"}
_ATTENTION_OPS = frozenset({"AttentionPlugin", "TRT_Attention", "Attention"})

_DTYPE_FORMATS: dict[str, str] = {
    "FLOAT16": "fp16",
    "BFLOAT16": "bf16",
    "FLOAT": "fp32",
    "DOUBLE": "fp64",
    "FLOAT8E4M3FN": "fp8",
    "FLOAT8E4M3FNUZ": "fp8",
    "FLOAT8E5M2": "fp8",
    "FLOAT8E5M2FNUZ": "fp8",
    "INT8": "int8",
    "UINT8": "int8",
    "INT4": "int4",
    "UINT4": "int4",
    "FLOAT4E2M1": "fp4",
}
_FORMAT_LABELS: dict[str, str] = {"int4_w4a16": "INT4 groupwise W4A16", "nvfp4_w4a16": "NVFP4 W4A16"}
_SHORT_FORMAT_LABELS: dict[str, str] = {"int4_w4a16": "int4", "nvfp4_w4a16": "nvfp4"}


def _int_attribute(attr: onnx.AttributeProto | None) -> int | None:
    from onnx import AttributeProto

    if attr is None or attr.type != AttributeProto.INT:
        return None
    return attr.i


def _dtype_format(data_type: int) -> str:
    from onnx import TensorProto

    try:
        name = TensorProto.DataType.Name(data_type)
    except ValueError:
        return "unknown"
    return _DTYPE_FORMATS.get(name, name.lower())


def _plugin_format(op_type: str) -> str:
    low = op_type.lower()
    if "int4" in low:
        return "int4_w4a16" if "groupwise" in low else "int4"
    if "fp4" in low:
        return "nvfp4_w4a16" if "a16" in low else "nvfp4"
    return next((fmt for fmt in ("fp8", "int8", "fp16") if fmt in low), op_type)


def _weight_sources(graph: onnx.GraphProto, external_tensors: set[str]) -> dict[str, tuple[tuple[int, ...], int]]:
    """name -> (dims, ONNX data type) of every tensor that can be a weight: initializers, Constant outputs and
    graph inputs the runtime fills from an external weight file (they must have a static shape)."""
    sources: dict[str, tuple[tuple[int, ...], int]] = {t.name: (tuple(t.dims), t.data_type) for t in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant" and node.output:
            for attr in node.attribute:
                if attr.name == "value" and attr.HasField("t"):
                    sources.setdefault(node.output[0], (tuple(attr.t.dims), attr.t.data_type))
    for value in graph.input:
        if value.name in external_tensors and value.name not in sources:
            tensor_type = value.type.tensor_type
            dims = tuple(d.dim_value if d.HasField("dim_value") else 0 for d in tensor_type.shape.dim)
            if dims and all(d > 0 for d in dims):
                sources[value.name] = (dims, tensor_type.elem_type)
    return sources


def _transposed(dims: tuple[int, ...], perms: list[list[int]]) -> tuple[int, ...] | None:
    """``dims`` after the Transposes met walking from the consumer back to the weight (so applied in reverse)."""
    shape = list(dims)
    for perm in reversed(perms):
        order = perm or list(range(len(shape)))[::-1]
        if sorted(order) != list(range(len(shape))):
            return None
        shape = [shape[i] for i in order]
    return tuple(shape)


def _resolve_weight(
    name: str,
    nodes: list[onnx.NodeProto],
    producers: dict[str, int],
    sources: dict[str, tuple[tuple[int, ...], int]],
) -> tuple[int, tuple[int, ...] | None, str] | None:
    """(params, shape as seen by the consumer or None, format) of the weight feeding ``name``; None if not a weight."""
    from onnx import AttributeProto

    perms: list[list[int]] = []
    reshaped = dequantized = False
    cast_to: int | None = None
    for _ in range(16):
        if name in sources:
            break
        index = producers.get(name)
        if index is None:
            return None
        node = nodes[index]
        if node.op_type not in _WEIGHT_PATH_OPS or not node.input:
            return None
        if node.op_type in _DEQUANTIZE_OPS:
            dequantized = True
        elif node.op_type in _RESHAPE_OPS:
            reshaped = True
        elif node.op_type == "Transpose":
            perms.append(next((list(a.ints) for a in node.attribute if a.name == "perm"), []))
        elif node.op_type == "Cast" and cast_to is None:
            cast_to = next((a.i for a in node.attribute if a.name == "to" and a.type == AttributeProto.INT), None)
        name = node.input[0]
    else:
        return None
    dims, data_type = sources[name]
    if len(dims) < 2 or any(d <= 0 for d in dims):
        return None
    if not dequantized and cast_to is not None:
        data_type = cast_to  # the engine folds the Cast and stores the converted weight
    shape = None if reshaped else _transposed(dims, perms)
    return math.prod(dims), shape, _dtype_format(data_type)


_GATE_PROJ = re.compile(r"layers\.(\d+)\.mlp\.gate_proj\b")


def _mlp_widths(nodes: list[onnx.NodeProto], producers: dict[str, int], linears: list[_Linear],
                num_layers: int) -> tuple[int, ...] | None:
    """Every layer's MLP width (its gate projection's output size), found by the weights' names; None when the
    graph does not name them that way."""
    widths: dict[int, int] = {}
    for linear in linears:
        node = nodes[linear.index]
        if len(node.input) < 2 or linear.out_dim is None:
            continue
        name = node.input[1]
        for _ in range(16):  # through Transpose / Cast / DequantizeLinear nodes to the weight itself
            match = _GATE_PROJ.search(name)
            if match:
                widths.setdefault(int(match.group(1)), linear.out_dim)
                break
            producer = producers.get(name)
            if producer is None or not nodes[producer].input:
                break
            name = nodes[producer].input[0]
    if sorted(widths) != list(range(num_layers)):
        return None
    return tuple(widths[i] for i in range(num_layers))


def _producers(nodes: list[onnx.NodeProto]) -> dict[str, int]:
    """Tensor name -> index of the node that produces it."""
    return {output: i for i, node in enumerate(nodes) for output in node.output if output}


def _linear_layers(
    graph: onnx.GraphProto, nodes: list[onnx.NodeProto], producers: dict[str, int], external_tensors: set[str]
) -> tuple[list[_Linear], str | None]:
    """(linear layers, problem) where problem says why some linear layer could not be measured."""
    sources = _weight_sources(graph, external_tensors)
    linears: list[_Linear] = []
    problem = None
    for index, node in enumerate(nodes):
        attrs = {attr.name: attr for attr in node.attribute}
        if "gemm_n" in attrs or "gemm_k" in attrs or node.op_type.startswith("Int4GroupwiseGemmPlugin"):
            gemm_n = _int_attribute(attrs.get("gemm_n"))
            gemm_k = _int_attribute(attrs.get("gemm_k"))
            if gemm_n is None or gemm_k is None or gemm_n <= 0 or gemm_k <= 0:
                problem = problem or f"{node.op_type} node {node.name or index!r} has no valid gemm_n/gemm_k attributes"
                continue
            group_size = _int_attribute(attrs.get("group_size"))
            linears.append(_Linear(index, gemm_n * gemm_k, gemm_n, _plugin_format(node.op_type), group_size))
        elif node.op_type in ("Gemm", "MatMul") and len(node.input) >= 2:
            weight = _resolve_weight(node.input[1], nodes, producers, sources)
            if weight is None:
                continue
            params, shape, fmt = weight
            out_dim = None
            if shape is not None and node.op_type == "MatMul":
                out_dim = shape[-1]
            elif shape is not None and len(shape) == 2:
                out_dim = shape[0] if _int_attribute(attrs.get("transB")) else shape[1]
            linears.append(_Linear(index, params, out_dim, fmt,
                                   zero_point_correction=_feeds_from(node.input[1], "zp_correction", nodes, producers)))
    return linears, problem


def _feeds_from(name: str, marker: str, nodes: list[onnx.NodeProto], producers: dict[str, int]) -> bool:
    """Whether tensor ``name``, or the weight it is converted from (through Cast / Transpose / ...), is named
    with ``marker``."""
    for _ in range(16):
        if marker in name:
            return True
        index = producers.get(name)
        if index is None or not nodes[index].input:
            return False
        name = nodes[index].input[0]
    return False


def _find_lm_head(
    nodes: list[onnx.NodeProto],
    producers: dict[str, int],
    output_names: list[str],
    linears: list[_Linear],
    output_vocab_size: int,
) -> _Linear | None:
    """The linear layer with output_vocab_size outputs; among several, the one nearest the ``logits`` output."""
    candidates = [linear for linear in linears if linear.out_dim == output_vocab_size]
    if len(candidates) <= 1:
        return candidates[0] if candidates else None
    by_index = {linear.index: linear for linear in candidates}
    target = "logits" if "logits" in output_names else (output_names[0] if output_names else "")
    start = producers.get(target)
    if start is not None:
        seen = {start}
        queue = deque([start])
        while queue:
            index = queue.popleft()
            if index in by_index:
                return by_index[index]
            for name in nodes[index].input:
                producer = producers.get(name)
                if producer is not None and producer not in seen:
                    seen.add(producer)
                    queue.append(producer)
    return candidates[-1]


def _format_label(fmt: str, group_sizes: set[int | None]) -> str:
    """Display name of a weight format; plugin op names without a known format are shown as they are."""
    label = _FORMAT_LABELS.get(fmt) or (fmt.upper() if fmt.islower() else fmt)
    known = sorted(size for size in group_sizes if size is not None)
    if fmt == "int4_w4a16" and known and len(known) == len(group_sizes):
        label += f" (group {'/'.join(str(size) for size in known)})"
    return label


def _describe_quantization(linears: list[_Linear], lm_head: _Linear | None) -> str:
    body: dict[str, int] = {}
    group_sizes: set[int | None] = set()
    for linear in linears:
        if linear is lm_head or linear.zero_point_correction:
            continue
        body[linear.fmt] = body.get(linear.fmt, 0) + linear.params
        if linear.fmt == "int4_w4a16":
            group_sizes.add(linear.group_size)
    if not body:
        if lm_head is None:
            return "unknown"
        body = {lm_head.fmt: lm_head.params}
        group_sizes = {lm_head.group_size}
        lm_head = None
    formats = sorted(body, key=lambda fmt: (-body[fmt], fmt))
    text = " + ".join(_format_label(fmt, group_sizes) for fmt in formats)
    if lm_head is not None and formats != [lm_head.fmt]:
        text += f", {_SHORT_FORMAT_LABELS.get(lm_head.fmt, lm_head.fmt)} lm_head"
    return text


# --------------------------------------------------------------------------- cost model availability

_NON_DENSE_MODEL_TYPES = frozenset({"qwen3_moe", "qwen3_5", "qwen3_5_moe", "nemotron_h", "gemma4", "gemma4_unified"})
# Matched in lower-cased text with punctuation removed ("linear_attn" -> "linearattn").
_HYBRID_SUBSTRINGS = ("mamba", "deltanet", "linearattention", "linearattn", "recurrentstate")
# Matched as whole words only, so that e.g. "classmate" does not count as "ssm".
_HYBRID_WORDS = frozenset({"ssm", "ssms", "gdn"})
_SPARSE_ATTENTION_WORDS = frozenset({"qsa", "sparse"})  # in attention op names (QsaAttentionPlugin)
_CAMEL_LOWER_UPPER = re.compile(r"([a-z0-9])([A-Z])")
_CAMEL_ACRONYM = re.compile(r"([A-Z]+)([A-Z][a-z])")


def _words(text: str) -> list[str]:
    """Lower-case words of snake_case, CamelCase or dotted text ("Int4MoePlugin" -> int, moe, plugin)."""
    spaced = _CAMEL_ACRONYM.sub(r"\1 \2", _CAMEL_LOWER_UPPER.sub(r"\1 \2", text))
    return re.findall(r"[a-z]+", spaced.lower())


def _mentions_hybrid(text: str) -> bool:
    compact = re.sub(r"[^a-z0-9]", "", text.lower())
    return any(marker in compact for marker in _HYBRID_SUBSTRINGS) or not _HYBRID_WORDS.isdisjoint(_words(text))


def _enabled(value: object) -> bool:
    """False for the "feature off" values exports write (0, false, null, "", [], {}, "none")."""
    if isinstance(value, str):
        return value.strip().lower() not in ("", "none", "null", "false", "0")
    return bool(value)


def _config_entries(obj: object, key: str = "") -> Iterator[tuple[str, object, bool]]:
    """(key, value, True) for every dict item and (enclosing key, string, False) for every string, recursively."""
    if isinstance(obj, dict):
        for name, value in obj.items():
            yield str(name), value, True
            yield from _config_entries(value, str(name))
    elif isinstance(obj, list):
        for value in obj:
            yield from _config_entries(value, key)
    elif isinstance(obj, str):
        yield key, obj, False


def _config_blocker(config: dict, model_type: str, num_layers: int) -> str | None:
    family = model_type.lower().removesuffix("_text")  # exports name the text model, e.g. "qwen3_5_text"
    if family in _NON_DENSE_MODEL_TYPES:
        return f"model type {model_type!r} is not a dense transformer (mixture-of-experts or hybrid layers)"
    for key, value, is_key in _config_entries(config):
        if is_key:
            if not _enabled(value):
                continue
            if "expert" in key.lower():
                return f"config.json key {key!r} indicates a mixture-of-experts model"
            if _mentions_hybrid(key):
                return f"config.json key {key!r} indicates state-space / linear-attention layers"
        elif isinstance(value, str) and _mentions_hybrid(value):
            return f"config.json value {value!r:.60} under {key!r} indicates state-space / linear-attention layers"
    attention_layers = config.get("num_attention_layers")
    if _is_count(attention_layers) and attention_layers != num_layers:
        return (
            f"config.json num_attention_layers is {attention_layers} but num_hidden_layers is {num_layers}, "
            "so not every layer is an attention layer"
        )
    window = config.get("sliding_window_size")  # written by the exporter only when the window is active
    if _window_binds(window, config):
        return _sliding_window_reason(f"config.json sliding_window_size is {window}")
    return None


def _window_binds(window: object, config: dict) -> bool:
    """True for a sliding window shorter than the longest context the model supports."""
    longest = config.get("max_position_embeddings")
    return _is_positive_int(window) and not (_is_positive_int(longest) and window >= longest)


def _sliding_window_reason(source: str) -> str:
    return (
        f"{source}: with sliding-window attention a token attends to (and reads the KV cache of) at most the "
        "window, which the dense formulas do not model"
    )


def _graph_blocker(
    nodes: list[onnx.NodeProto],
    linears: list[_Linear],
    lm_head: _Linear | None,
    problem: str | None,
    expected_attention: dict[str, int],
    config: dict,
    num_layers: int,
    output_vocab_size: int,
) -> str | None:
    for node in nodes:
        op_type = node.op_type
        # A substring test: camel-case splitting cannot see the word in e.g. "NvFP4MoEPluginGeforce".
        if "moe" in op_type.lower():
            return f"model.onnx contains mixture-of-experts layers ({op_type})"
        if _mentions_hybrid(op_type) or "conv1d" in op_type.lower():  # causal_conv1d: Mamba / GDN
            return f"model.onnx contains state-space / linear-attention layers ({op_type})"
        words = _words(op_type)
        if "attention" in words and not _SPARSE_ATTENTION_WORDS.isdisjoint(words):  # QsaAttentionPlugin
            return f"model.onnx contains sparse attention ({op_type}), whose cost depends on the selected keys"
    if problem:
        return problem
    if not linears:
        return "model.onnx contains no linear layers the cost model can read"
    if lm_head is None:
        return f"no linear layer in model.onnx has {output_vocab_size} outputs, so the lm_head cannot be identified"
    attention = [node for node in nodes if node.op_type in _ATTENTION_OPS]
    if attention and len(attention) != num_layers:
        kinds = ", ".join(sorted({node.op_type for node in attention}))
        return (
            f"model.onnx has {len(attention)} attention node(s) ({kinds}) but config.json num_hidden_layers "
            f"is {num_layers}"
        )
    for node in attention:
        for attr in node.attribute:
            value = _int_attribute(attr)
            if attr.name == "sliding_window_size" and _window_binds(value, config):
                return _sliding_window_reason(f"model.onnx attention sliding_window_size is {value}")
            if attr.name in expected_attention and value is not None and value != expected_attention[attr.name]:
                return (
                    f"model.onnx attention {attr.name}={value} does not match config.json "
                    f"({expected_attention[attr.name]})"
                )
    return None


# --------------------------------------------------------------------------- public API

_EXPORT_COMPONENT = "llm"  # tensorrt-edgellm-export writes the language model into <output_dir>/llm


def model_name(path: Path) -> str:
    """The name a model folder is served under: the folder's name, except that the exporter's ``llm`` folder
    (``<export>/llm``) takes the export's name, so that every export is not served as "llm"."""
    root = Path(os.path.abspath(path))
    if root.name == _EXPORT_COMPONENT and root.parent.name:
        return root.parent.name
    return root.name


def load_model(path: Path) -> ModelInfo:
    """Validate a model folder and read its architecture and cost facts.

    Raises ModelError (with an actionable message) when: the path is not a directory; config.json is missing
    or not JSON; config["engine_role"] is present and not "llm"; config["spec_decode_type"] is present and not
    "none" (speculative decoding is not supported); model.onnx, embedding.safetensors, tokenizer.json or
    tokenizer_config.json is missing; an external weight file listed in config is missing; reduced_vocab_size is
    set but vocab_map.safetensors is missing; model.onnx references external data that is missing.

    Parses model.onnx with ``onnx.load(..., load_external_data=False)``. Linear layers are
    ``Int4GroupwiseGemmPluginV2`` nodes (params = gemm_n * gemm_k attributes) plus ``Gemm`` / ``MatMul`` nodes
    whose weight input is an initializer (params = product of its dims). The lm_head is the linear layer whose
    output dimension equals output_vocab_size (it feeds the ``logits`` graph output). cost_model is
    "dense_transformer" unless config shows experts (any key containing "expert"), state-space / linear-attention
    layers (keys or values mentioning "mamba", "ssm", "linear_attention", "deltanet"), or model_type is one of
    qwen3_moe, qwen3_5, qwen3_5_moe, nemotron_h, gemma4, gemma4_unified; then "unavailable" with a reason.
    """
    root = Path(os.path.abspath(path))
    try:
        return _load_folder(root)
    except OSError as exc:  # e.g. EACCES on a parent folder: pathlib only maps "not found" errors to False
        raise ModelError(
            f"cannot read the model folder {root} ({exc}); check that the user running edgeserve can read it"
        ) from exc


def _load_folder(root: Path) -> ModelInfo:
    config = _read_config(root)
    _check_engine_kind(root, config)
    _check_required_files(root)
    external = _external_weight_files(root, config)
    reduced_vocab = _reduced_vocab_size(root, config)
    generation = _read_generation_config(root)

    model_type = config.get("model")
    if not isinstance(model_type, str) or not model_type.strip():
        raise ModelError(
            f"config.json has no 'model' architecture name (found {model_type!r}); re-export the model with "
            "tensorrt-edgellm-export"
        )
    num_layers = _config_int(config, "num_hidden_layers")
    hidden_size = _config_int(config, "hidden_size")
    num_heads = _config_int(config, "num_attention_heads")
    num_kv_heads = _config_int(config, "num_key_value_heads")
    head_dim = _config_int(config, "head_dim")
    vocab_size = _config_int(config, "vocab_size")
    if reduced_vocab is not None and reduced_vocab > vocab_size:
        raise ModelError(
            f"config.json: reduced_vocab_size ({reduced_vocab}) is larger than vocab_size ({vocab_size}); "
            "run tensorrt-edgellm-reduce-vocab again or export the model with its full vocabulary"
        )
    output_vocab_size = reduced_vocab or vocab_size
    kv_cache_dtype, kv_dtype_bytes = _kv_cache_dtype(config)
    edgellm_version = config.get("edgellm_version")

    model = _load_onnx(root / "model.onnx")
    onnx_data_files, onnx_data_bytes = _check_external_data(root, model)
    external_tensors = {name for entry in external for name in entry.tensors}
    nodes = list(model.graph.node)
    producers = _producers(nodes)
    linears, problem = _linear_layers(model.graph, nodes, producers, external_tensors)
    output_names = [value.name for value in model.graph.output]
    lm_head = _find_lm_head(nodes, producers, output_names, linears, output_vocab_size)
    expected_attention = {"num_q_heads": num_heads, "num_kv_heads": num_kv_heads, "head_size": head_dim}
    reason = _config_blocker(config, model_type, num_layers) or _graph_blocker(
        nodes, linears, lm_head, problem, expected_attention, config, num_layers, output_vocab_size
    )

    # Weights live in model.onnx's external data files (or in model.onnx itself) plus the external weight files.
    # Weights live in model.onnx's external data (the bytes it references) or in model.onnx itself, plus the
    # external weight files.
    streamed = onnx_data_bytes if onnx_data_files else (root / "model.onnx").stat().st_size
    streamed += sum((root / name).stat().st_size for name in {entry.file for entry in external})
    resident = streamed + (root / _EMBEDDING).stat().st_size
    if reduced_vocab:
        resident += (root / _VOCAB_MAP).stat().st_size
    chat_template = root / _CHAT_TEMPLATE
    file_bytes = {rel: file.stat().st_size for rel, file in _list_files(root)}

    return ModelInfo(
        name=model_name(root),
        path=root,
        model_type=model_type,
        edgellm_version=None if edgellm_version is None else str(edgellm_version),
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        output_vocab_size=output_vocab_size,
        kv_cache_dtype=kv_cache_dtype,
        kv_bytes_per_token=_kv_bytes_per_token(config, num_layers, num_kv_heads, head_dim, kv_dtype_bytes),
        eos_token_ids=_eos_token_ids(config, generation),
        quantization=_describe_quantization(linears, lm_head),
        linear_params=sum(linear.params for linear in linears),
        lm_head_params=lm_head.params if lm_head is not None else 0,
        embedding_params=vocab_size * hidden_size,
        resident_weight_bytes=resident,
        streamed_weight_bytes=streamed,
        file_bytes=file_bytes,
        has_chat_template=chat_template.is_file() and chat_template.stat().st_size > 0,
        generation_defaults=generation,
        cost_model=_UNAVAILABLE if reason else _DENSE,
        cost_model_reason=reason,
        mlp_widths=_mlp_widths(nodes, producers, linears, num_layers),
    )


_FULL_HASH_LIMIT = 1 << 20  # 1 MiB
_SAMPLE_BYTES = 64 << 10  # 64 KiB
_FINGERPRINT_TAG = b"edgeserve.model-fingerprint.v2\n"


def _affects_engine(rel: str) -> bool:
    """Files the engine build never reads: sampling defaults (read by the server), documentation, and hidden files
    and folders (e.g. the .cache/ that ``hf download --local-dir`` leaves)."""
    parts = rel.split("/")
    if any(part.startswith(".") for part in parts):
        return False
    name = parts[-1]
    upper = name.upper()
    return not (name == "generation_config.json" or upper.endswith(".MD")
                or upper.startswith(("LICENSE", "NOTICE")))


def _raise(error: OSError) -> None:
    raise error


def _regular_files(root: Path) -> list[tuple[str, Path]]:
    """(POSIX path relative to root, path) of every regular file below root (file symlinks followed), sorted.

    OSError when a folder cannot be listed: a silently incomplete listing would let a changed model keep its
    fingerprint (files can still be opened by name, so the build would use them) and reuse a stale engine.
    """
    found = []
    for directory, _, filenames in os.walk(root, onerror=_raise):
        for filename in filenames:
            file = Path(directory, filename)
            if file.is_file():
                found.append((file.relative_to(root).as_posix(), file))
    found.sort(key=lambda item: item[0])
    return found


def _list_files(root: Path) -> list[tuple[str, Path]]:
    try:
        return _regular_files(root)
    except OSError as exc:
        raise ModelError(f"cannot list the files in {root} ({exc}); make the folder readable") from exc


def _file_digest(file: Path, size: int) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as handle:
        if size <= _FULL_HASH_LIMIT:
            for chunk in iter(lambda: handle.read(1 << 16), b""):
                digest.update(chunk)
        else:
            for offset in (0, (size - _SAMPLE_BYTES) // 2, size - _SAMPLE_BYTES):
                handle.seek(offset)
                digest.update(handle.read(_SAMPLE_BYTES))
    return digest.hexdigest()


def fingerprint(path: Path) -> str:
    """Stable sha256 hex digest of a model folder's contents, independent of its location and mtimes.

    Every regular file contributes its relative name and size; files <= 1 MiB are hashed in full; larger files
    contribute three 64 KiB samples (start, middle, end). Files are visited in sorted order. Used in the engine
    cache key, so editing or replacing any model file triggers a rebuild — except files the engine build never
    reads (generation_config.json, *.md, LICENSE*/NOTICE*, hidden files and folders), so changing sampling
    defaults or docs does not.
    """
    root = Path(path)
    try:
        is_dir = root.is_dir()
    except OSError as exc:  # pathlib only maps "not found" errors to False, not e.g. EACCES on a parent
        raise ModelError(f"cannot access {root} ({exc})") from exc
    if not is_dir:
        raise ModelError(f"{root} is not a directory")
    digest = hashlib.sha256(_FINGERPRINT_TAG)
    for rel, file in _list_files(root):
        if not _affects_engine(rel):
            continue
        try:
            size = file.stat().st_size
            file_digest = _file_digest(file, size)
        except OSError as exc:
            raise ModelError(f"cannot read {file} ({exc})") from exc
        digest.update(os.fsencode(rel) + f"\0{size}\0{file_digest}\n".encode())
    return digest.hexdigest()


_KV_PAGE_TOKENS = 128
_RUNTIME_BYTES = 300 << 20  # 300 MiB
_ACTIVATION_FACTOR = 20


def estimate_memory(model: ModelInfo, max_batch_size: int, kv_capacity: int, max_input_len: int) -> dict[str, int]:
    """Estimated device memory for serving, in bytes.

    Keys: weights (resident_weight_bytes), kv_cache (max_batch_size * ceil(kv_capacity / 128) * 128 *
    kv_bytes_per_token, Edge-LLM's paged pool), activations (max_batch_size * max_input_len * hidden_size * 2 * 20,
    a conservative TensorRT activation estimate), runtime (300 MiB: CUDA context, TensorRT, plugins, tokenizer),
    total (sum). These are planning estimates; the service logs measured values after loading.
    """
    max_batch_size = _count("max_batch_size", max_batch_size)
    kv_capacity = _count("kv_capacity", kv_capacity)
    max_input_len = _count("max_input_len", max_input_len)
    pages = -(-kv_capacity // _KV_PAGE_TOKENS)
    estimate = {
        "weights": model.resident_weight_bytes,
        "kv_cache": max_batch_size * pages * _KV_PAGE_TOKENS * model.kv_bytes_per_token,
        "activations": max_batch_size * max_input_len * model.hidden_size * 2 * _ACTIVATION_FACTOR,
        "runtime": _RUNTIME_BYTES,
    }
    estimate["total"] = sum(estimate.values())
    return estimate


__all__ = ["ModelError", "ModelInfo", "estimate_memory", "fingerprint", "load_model", "model_name"]
