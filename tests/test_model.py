"""Tests for edgeserve.model: folder validation, graph-derived cost facts, fingerprints and memory estimates."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import struct
from dataclasses import replace
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from edgeserve.model import ModelError, ModelInfo, estimate_memory, fingerprint, load_model

REPO = Path(__file__).resolve().parents[1]
FULL = REPO / "models" / "full_vocab_model"
REDUCED = REPO / "models" / "reduced_vocab_model"

HIDDEN, VOCAB, LAYERS, HEADS, KV_HEADS, HEAD_DIM = 64, 256, 2, 4, 2, 16
WIDTHS = (96, 80)  # per-layer pruned FFN widths, deliberately different from config["intermediate_size"]
GROUP = 32
KV_WIDTH = KV_HEADS * HEAD_DIM
ATTN_PARAMS = LAYERS * (2 * HIDDEN * HIDDEN + 2 * KV_WIDTH * HIDDEN)  # q, o + k, v
FFN_PARAMS = sum(3 * width * HIDDEN for width in WIDTHS)  # gate, up, down
KV_BYTES = 2 * LAYERS * KV_HEADS * HEAD_DIM * 2
ATTN_FLOPS_PER_PAIR = 4 * LAYERS * HEADS * HEAD_DIM
FFN_FILE = "external_int4_ffn_weights.safetensors"
LM_HEAD_FILE = "external_lm_head_weight.safetensors"
LM_HEAD_INPUT = "_model.lm_head.weight"
DROP = object()  # config override value that removes the key
MIB = 1 << 20


# --------------------------------------------------------------------------- synthetic model folders

def _raw(name: str, data_type: int, dims: list[int], byte: bytes) -> onnx.TensorProto:
    itemsize = {TensorProto.FLOAT16: 2, TensorProto.INT8: 1}[data_type]
    return helper.make_tensor(name, data_type, dims, byte * (itemsize * math.prod(dims)), raw=True)


def _int64(name: str, values: list[int]) -> onnx.TensorProto:
    data = struct.pack(f"<{len(values)}q", *values)
    return helper.make_tensor(name, TensorProto.INT64, [len(values)], data, raw=True)


def _projections() -> list[tuple[str, int, int, bool]]:
    """(name, gemm_n, gemm_k, externalized) of every linear layer except the lm_head, in graph order."""
    layers = []
    for i, width in enumerate(WIDTHS):
        prefix = f"_model.model.layers.{i}"
        layers += [
            (f"{prefix}.self_attn.q_proj", HIDDEN, HIDDEN, False),
            (f"{prefix}.self_attn.k_proj", KV_WIDTH, HIDDEN, False),
            (f"{prefix}.self_attn.v_proj", KV_WIDTH, HIDDEN, False),
            (f"{prefix}.self_attn.o_proj", HIDDEN, HIDDEN, False),
            (f"{prefix}.mlp.gate_proj", width, HIDDEN, True),
            (f"{prefix}.mlp.up_proj", width, HIDDEN, True),
            (f"{prefix}.mlp.down_proj", HIDDEN, width, True),
        ]
    return layers


def _ffn_tensor_names() -> list[str]:
    return [f"{name}.{part}" for name, _, _, ext in _projections() if ext for part in ("qweight", "scales")]


def _build_graph(
    *, fp16: bool, group: int | None, lm_rows: int, lm_head: str, attention: int, attention_attrs: dict, extra_nodes
) -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    inputs = [helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT16, ["physical_tokens", HIDDEN])]
    prev = "inputs_embeds"
    for index, (name, n, k, external) in enumerate(_projections()):
        out = f"h{index}"
        if fp16:
            inits.append(_raw(f"{name}.weight", TensorProto.FLOAT16, [k, n], b"\x00"))
            nodes.append(helper.make_node("MatMul", [prev, f"{name}.weight"], [out], name=f"mm{index}"))
        else:
            shapes = {
                "qweight": (TensorProto.INT8, [k // 2, n]),
                "scales": (TensorProto.FLOAT16, [max(k // GROUP, 1), n]),
            }
            for part, (data_type, dims) in shapes.items():
                if external:  # externalized FFN weights are graph inputs filled by the runtime
                    inputs.append(helper.make_tensor_value_info(f"{name}.{part}", data_type, dims))
                else:
                    inits.append(_raw(f"{name}.{part}", data_type, dims, b"\x01"))
            attrs = {"gemm_n": n, "gemm_k": k} | ({"group_size": group} if group is not None else {})
            nodes.append(
                helper.make_node(
                    "Int4GroupwiseGemmPluginV2",
                    [prev, f"{name}.qweight", f"{name}.scales"],
                    [out],
                    name=f"n{index}",
                    domain="trt_edgellm",
                    **attrs,
                )
            )
        prev = out
    head_attrs = {"num_q_heads": HEADS, "num_kv_heads": KV_HEADS, "head_size": HEAD_DIM} | attention_attrs
    for i in range(attention):
        nodes.append(
            helper.make_node("AttentionPlugin", ["inputs_embeds"], [f"attn{i}"], domain="trt_edgellm", **head_attrs)
        )
    # Not linear layers: a Gather with an INT64 permutation initializer and an activation x activation MatMul.
    inits.append(_int64("perm", list(range(HIDDEN))[::-1]))
    nodes += [
        helper.make_node("Gather", [prev, "perm"], ["permuted"], axis=1),
        helper.make_node("Transpose", ["permuted"], ["permuted_t"]),
        helper.make_node("MatMul", ["permuted", "permuted_t"], ["scores"]),
    ]
    if lm_head == "gemm":
        inits.append(_raw(LM_HEAD_INPUT, TensorProto.FLOAT16, [lm_rows, HIDDEN], b"\x00"))
        nodes.append(helper.make_node("Gemm", ["permuted", LM_HEAD_INPUT], ["linear"], transB=1))
    elif lm_head == "external":
        inputs.append(helper.make_tensor_value_info(LM_HEAD_INPUT, TensorProto.FLOAT16, [lm_rows, HIDDEN]))
        nodes.append(helper.make_node("Gemm", ["permuted", LM_HEAD_INPUT], ["linear"], transB=1))
    elif lm_head == "matmul":
        inits.append(_raw(LM_HEAD_INPUT, TensorProto.FLOAT16, [HIDDEN, lm_rows], b"\x00"))
        nodes.append(helper.make_node("MatMul", ["permuted", LM_HEAD_INPUT], ["linear"]))
    elif lm_head == "int8_dq":  # [V, H] int8 -> DequantizeLinear -> Transpose -> MatMul, so the consumer sees [H, V]
        inits += [
            _raw(LM_HEAD_INPUT, TensorProto.INT8, [lm_rows, HIDDEN], b"\x01"),
            _raw("lm_scale", TensorProto.FLOAT16, [], b"\x00"),
        ]
        nodes += [
            helper.make_node("DequantizeLinear", [LM_HEAD_INPUT, "lm_scale"], ["lm_dq"]),
            helper.make_node("Transpose", ["lm_dq"], ["lm_t"]),
            helper.make_node("MatMul", ["permuted", "lm_t"], ["linear"]),
        ]
    inits.append(_int64("logits_shape", [-1, lm_rows]))
    nodes += [
        helper.make_node("Cast", ["linear"], ["logits_f32"], to=TensorProto.FLOAT),
        helper.make_node("Reshape", ["logits_f32", "logits_shape"], ["logits"]),
        *extra_nodes,
    ]
    graph = helper.make_graph(
        nodes,
        "llm",
        inputs,
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["num_sequences", lm_rows])],
        initializer=inits,
    )
    opsets = [helper.make_opsetid("", 17), helper.make_opsetid("trt_edgellm", 1)]
    return helper.make_model(graph, opset_imports=opsets, producer_name="test")


def make_model(
    root: Path,
    *,
    fp16: bool = False,
    group: int | None = GROUP,
    reduced: int | None = None,
    lm_rows: int | None = None,
    lm_head: str | None = None,
    attention: int = 0,
    attention_attrs: dict | None = None,
    extra_nodes: tuple = (),
    external_data: bool = True,
    config: dict | None = None,
    chat_template: str | None = "{{ messages }}",
    generation: dict | str | None = None,
) -> Path:
    """Write a small Edge-LLM-style export folder (2 layers, pruned FFN widths, INT4 plugin nodes by default)."""
    root.mkdir(parents=True)
    lm_head = lm_head or ("matmul" if fp16 else "gemm")
    rows = lm_rows if lm_rows is not None else (reduced or VOCAB)
    model = _build_graph(
        fp16=fp16,
        group=group,
        lm_rows=rows,
        lm_head=lm_head,
        attention=attention,
        attention_attrs=attention_attrs or {},
        extra_nodes=extra_nodes,
    )
    if external_data:
        onnx.save_model(
            model,
            str(root / "model.onnx"),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.onnx.data",
            size_threshold=64,
        )
    else:
        onnx.save_model(model, str(root / "model.onnx"))

    cfg: dict = {
        "model": "llama",
        "spec_decode_type": "none",
        "engine_role": "llm",
        "edgellm_version": "0.11.0",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": 128,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "rope_scaling": {"rope_type": "llama3", "factor": 8.0},
        "num_deepstack_features": 0,
        "use_vision_bidirectional_attention": False,
        "ple_enabled": False,
        "kv_cache_dtype": "fp16",
        "eos_token_id": [1, 2],
    }
    files = []
    if not fp16:
        files.append({"file": FFN_FILE, "kind": "int4_ffn_weights", "tensors": _ffn_tensor_names()})
        (root / FFN_FILE).write_bytes(b"\x01" * 4000)
    if lm_head == "external":
        files.append({"file": LM_HEAD_FILE, "kind": "lm_head_weight", "tensors": [LM_HEAD_INPUT]})
        (root / LM_HEAD_FILE).write_bytes(b"\x02" * (rows * HIDDEN * 2))
    if files:
        cfg["external_weight_files"] = files
    if reduced is not None:
        cfg["reduced_vocab_size"] = reduced
        (root / "vocab_map.safetensors").write_bytes(b"\x03" * (reduced * 4))
    for key, value in (config or {}).items():
        if value is DROP:
            cfg.pop(key, None)
        else:
            cfg[key] = value
    (root / "config.json").write_text(json.dumps(cfg, indent=2))
    (root / "embedding.safetensors").write_bytes(b"\x04" * (VOCAB * HIDDEN * 2))
    (root / "tokenizer.json").write_text("{}")
    (root / "tokenizer_config.json").write_text("{}")
    if chat_template is not None:
        (root / "chat_template.jinja").write_text(chat_template)
    if generation is not None:
        text = generation if isinstance(generation, str) else json.dumps(generation)
        (root / "generation_config.json").write_text(text)
    return root


def size(path: Path) -> int:
    return path.stat().st_size


def _plugin_node(op_type: str) -> dict:
    """make_model kwargs adding one trt_edgellm plugin node of ``op_type``."""
    return {"extra_nodes": (helper.make_node(op_type, ["inputs_embeds"], [op_type.lower()], domain="trt_edgellm"),)}


@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    return make_model(tmp_path / "tiny-llama")


# --------------------------------------------------------------------------- load_model: facts

def test_loads_synthetic_int4_model(model_dir: Path) -> None:
    info = load_model(model_dir)

    assert isinstance(info, ModelInfo)
    assert info.name == "tiny-llama"
    assert info.path == model_dir
    assert info.model_type == "llama"
    assert info.edgellm_version == "0.11.0"
    assert (info.num_layers, info.hidden_size, info.num_heads, info.num_kv_heads, info.head_dim) == (
        LAYERS,
        HIDDEN,
        HEADS,
        KV_HEADS,
        HEAD_DIM,
    )
    assert info.vocab_size == info.output_vocab_size == VOCAB
    assert info.kv_cache_dtype == "fp16"
    assert info.kv_bytes_per_token == KV_BYTES
    assert info.eos_token_ids == (1, 2)
    # Widths come from the plugin attributes, not config["intermediate_size"]; the Gather and the
    # activation x activation MatMul are not counted.
    assert info.lm_head_params == VOCAB * HIDDEN
    assert info.linear_params == ATTN_PARAMS + FFN_PARAMS + VOCAB * HIDDEN
    assert info.embedding_params == VOCAB * HIDDEN
    assert info.quantization == f"INT4 groupwise W4A16 (group {GROUP}), fp16 lm_head"
    assert info.streamed_weight_bytes == size(model_dir / "model.onnx.data") + size(model_dir / FFN_FILE)
    assert info.resident_weight_bytes == info.streamed_weight_bytes + size(model_dir / "embedding.safetensors")
    assert info.has_chat_template is True
    assert info.generation_defaults == {}
    assert info.cost_model == "dense_transformer"
    assert info.cost_model_reason is None
    names = sorted(p.name for p in model_dir.iterdir())
    assert sorted(info.file_bytes) == names
    assert all(info.file_bytes[name] == size(model_dir / name) for name in names)


def test_fp16_model_with_matmul_weights(tmp_path: Path) -> None:
    info = load_model(make_model(tmp_path / "m", fp16=True))
    assert info.quantization == "FP16"
    assert info.linear_params == ATTN_PARAMS + FFN_PARAMS + VOCAB * HIDDEN
    assert info.lm_head_params == VOCAB * HIDDEN  # MatMul weight [hidden, vocab]: output dim is the last one
    assert info.cost_model == "dense_transformer"


def test_quantization_without_group_size_attribute(tmp_path: Path) -> None:
    info = load_model(make_model(tmp_path / "m", group=None))
    assert info.quantization == "INT4 groupwise W4A16, fp16 lm_head"


def test_dequantized_transposed_lm_head(tmp_path: Path) -> None:
    info = load_model(make_model(tmp_path / "m", fp16=True, lm_head="int8_dq"))
    assert info.lm_head_params == VOCAB * HIDDEN
    assert info.quantization == "FP16, int8 lm_head"
    assert info.cost_model == "dense_transformer"


def test_constant_node_weights_and_folded_casts(tmp_path: Path) -> None:
    fp32 = helper.make_tensor("c32", TensorProto.FLOAT, [HIDDEN, 4], [0.0] * (HIDDEN * 4))
    extra = (
        helper.make_node("Constant", [], ["c16"], value=_raw("c16", TensorProto.FLOAT16, [HIDDEN, 8], b"\x00")),
        helper.make_node("MatMul", ["inputs_embeds", "c16"], ["c16_out"]),
        helper.make_node("Constant", [], ["c32"], value=fp32),
        helper.make_node("Cast", ["c32"], ["c32_fp16"], to=TensorProto.FLOAT16),  # folded: stored as fp16
        helper.make_node("MatMul", ["inputs_embeds", "c32_fp16"], ["c32_out"]),
    )
    info = load_model(make_model(tmp_path / "m", fp16=True, extra_nodes=extra))
    assert info.linear_params == ATTN_PARAMS + FFN_PARAMS + VOCAB * HIDDEN + HIDDEN * 8 + HIDDEN * 4
    assert info.quantization == "FP16"


def test_externalized_lm_head_is_read_from_graph_input(tmp_path: Path) -> None:
    root = make_model(tmp_path / "m", lm_head="external")
    info = load_model(root)
    assert info.lm_head_params == VOCAB * HIDDEN
    assert info.linear_params == ATTN_PARAMS + FFN_PARAMS + VOCAB * HIDDEN
    assert info.streamed_weight_bytes == sum(size(root / f) for f in ("model.onnx.data", FFN_FILE, LM_HEAD_FILE))
    assert info.cost_model == "dense_transformer"


def test_inline_weights_are_streamed_from_model_onnx(tmp_path: Path) -> None:
    root = make_model(tmp_path / "m", external_data=False)
    info = load_model(root)
    assert not (root / "model.onnx.data").exists()
    assert info.streamed_weight_bytes == size(root / "model.onnx") + size(root / FFN_FILE)


def test_reduced_vocabulary(tmp_path: Path) -> None:
    root = make_model(tmp_path / "m", reduced=64)
    info = load_model(root)
    assert info.vocab_size == VOCAB
    assert info.output_vocab_size == 64
    assert info.lm_head_params == 64 * HIDDEN
    assert info.embedding_params == VOCAB * HIDDEN
    assert info.resident_weight_bytes == (
        info.streamed_weight_bytes + size(root / "embedding.safetensors") + size(root / "vocab_map.safetensors")
    )
    assert info.decode_bytes(1) == info.streamed_weight_bytes + KV_BYTES + 4 * 64


def test_reduced_vocab_size_zero_means_full_vocabulary(tmp_path: Path) -> None:
    info = load_model(make_model(tmp_path / "m", config={"reduced_vocab_size": 0}))
    assert info.output_vocab_size == VOCAB


def test_lm_head_is_the_candidate_nearest_the_logits(tmp_path: Path) -> None:
    # vocab == hidden: q/o/down projections also have `vocab` outputs; the Gemm feeding logits wins.
    root = make_model(tmp_path / "m", lm_rows=HIDDEN, config={"vocab_size": HIDDEN})
    info = load_model(root)
    assert info.lm_head_params == HIDDEN * HIDDEN
    assert info.quantization.endswith(", fp16 lm_head")
    assert info.cost_model == "dense_transformer"


def test_relative_and_str_paths(model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(model_dir.parent)
    info = load_model(Path("tiny-llama"))
    assert info.path == model_dir and info.path.is_absolute()
    assert load_model(str(model_dir)).name == "tiny-llama"  # type: ignore[arg-type]


def test_edgellm_version_is_reported_not_checked(tmp_path: Path) -> None:
    assert load_model(make_model(tmp_path / "a", config={"edgellm_version": "9.9.9"})).edgellm_version == "9.9.9"
    assert load_model(make_model(tmp_path / "b", config={"edgellm_version": DROP})).edgellm_version is None


@pytest.mark.parametrize(
    "key",
    [
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "kv_cache_dtype",
    ],
)
def test_fields_the_runtime_requires_have_no_defaults(tmp_path: Path, key: str) -> None:
    # The Edge-LLM runtime / builder read these without a default, so a folder lacking one cannot be served:
    # report it now instead of after a long engine build.
    with pytest.raises(ModelError, match=rf"config\.json has no {key}\b.*re-export"):
        load_model(make_model(tmp_path / "m", config={key: DROP}))


@pytest.mark.parametrize(("dtype", "nbytes"), [("fp16", 2), ("bf16", 2), ("fp8", 1), ("int8", 1), ("fp32", 4)])
def test_kv_cache_dtypes(tmp_path: Path, dtype: str, nbytes: int) -> None:
    info = load_model(make_model(tmp_path / "m", config={"kv_cache_dtype": dtype}))
    assert info.kv_cache_dtype == dtype
    assert info.kv_bytes_per_token == 2 * LAYERS * KV_HEADS * HEAD_DIM * nbytes


@pytest.mark.parametrize("dtype", ["FP16", "Fp8", " fp16", "fp4", 2])
def test_kv_cache_dtype_must_match_the_runtime_names_exactly(tmp_path: Path, dtype: object) -> None:
    # The runtime's parser is case-sensitive: "FP16" would only fail when the built engine is loaded.
    with pytest.raises(ModelError, match=r"kv_cache_dtype .* is not supported \(expected one of fp16, "):
        load_model(make_model(tmp_path / "m", config={"kv_cache_dtype": dtype}))


def test_kv_layer_configs_table_is_summed(tmp_path: Path) -> None:
    table = [{"num_kv_heads": 2, "head_dim": 16}, {"num_kv_heads": 1, "head_dim": 64}]
    info = load_model(make_model(tmp_path / "m", config={"kv_layer_configs": table}))
    assert info.kv_bytes_per_token == 2 * (2 * 16 + 1 * 64) * 2


def test_malformed_kv_layer_configs_fall_back_to_uniform_layers(tmp_path: Path) -> None:
    info = load_model(make_model(tmp_path / "m", config={"kv_layer_configs": [{"num_kv_heads": "x"}]}))
    assert info.kv_bytes_per_token == KV_BYTES
    info = load_model(make_model(tmp_path / "n", config={"kv_layer_configs": [{"num_kv_heads": 2}, 7]}))
    assert info.kv_bytes_per_token == KV_BYTES


@pytest.mark.parametrize(
    ("overrides", "kv_layers"),
    [
        # Table entries without dims take the global ones (as the builder does); null entries are recurrent layers.
        pytest.param(
            {"kv_layer_configs": [{"kv_cache_capacity": 64}, None], "num_linear_attn_layers": 1},
            1,
            id="table-defaults",
        ),
        pytest.param({"kv_layer_configs": [{}, {"head_dim": HEAD_DIM}]}, 2, id="table-empty-entries"),
        # Hybrid without a table: only the num_attention_layers attention layers have a KV cache.
        pytest.param({"num_linear_attn_layers": 1, "num_attention_layers": 1}, 1, id="hybrid-no-table"),
        # Without linear-attention layers the builder sizes every layer, whatever num_attention_layers says.
        pytest.param({"num_attention_layers": 1}, LAYERS, id="dense-num-attention-layers"),
        pytest.param({"num_linear_attn_layers": 1}, LAYERS, id="hybrid-without-attention-count"),
    ],
)
def test_kv_bytes_follow_the_builder_layout(tmp_path: Path, overrides: dict, kv_layers: int) -> None:
    info = load_model(make_model(tmp_path / "m", config=overrides))
    assert info.kv_bytes_per_token == 2 * kv_layers * KV_HEADS * HEAD_DIM * 2


@pytest.mark.parametrize(
    ("config_eos", "generation", "expected"),
    [
        (5, None, (5,)),
        ([1, 2], {"eos_token_id": [2, 3], "temperature": 0.6}, (1, 2, 3)),
        (DROP, {"eos_token_id": 7}, (7,)),
        (DROP, None, ()),
        ([1], {"eos_token_id": "bad"}, (1,)),
    ],
)
def test_eos_token_ids(tmp_path: Path, config_eos: object, generation: dict | None, expected: tuple) -> None:
    root = make_model(tmp_path / "m", config={"eos_token_id": config_eos}, generation=generation)
    assert load_model(root).eos_token_ids == expected


def test_generation_config_is_parsed(tmp_path: Path) -> None:
    defaults = {"temperature": 0.6, "top_p": 0.9, "do_sample": True}
    info = load_model(make_model(tmp_path / "m", generation=defaults))
    assert info.generation_defaults == defaults


@pytest.mark.parametrize(("template", "expected"), [("{{ messages }}", True), ("", False), (None, False)])
def test_has_chat_template(tmp_path: Path, template: str | None, expected: bool) -> None:
    assert load_model(make_model(tmp_path / "m", chat_template=template)).has_chat_template is expected


def test_as_dict_is_json_serialisable(tmp_path: Path) -> None:
    info = load_model(make_model(tmp_path / "m", generation={"temperature": 0.6}))
    data = info.as_dict()
    assert json.loads(json.dumps(data, allow_nan=False)) == data  # strict JSON, as Starlette renders it
    assert "file_bytes" not in data
    assert data["path"] == str(info.path)
    assert data["eos_token_ids"] == [1, 2]
    assert data["linear_params"] == info.linear_params
    assert data["cost_model"] == "dense_transformer" and data["cost_model_reason"] is None
    data["generation_defaults"]["temperature"] = 1.0
    assert info.generation_defaults == {"temperature": 0.6}


# --------------------------------------------------------------------------- load_model: validation errors

@pytest.mark.parametrize(
    "name",
    [
        "config.json",
        "model.onnx",
        "model.onnx.data",
        "embedding.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        FFN_FILE,
    ],
)
def test_missing_file_is_reported(model_dir: Path, name: str) -> None:
    (model_dir / name).unlink()
    with pytest.raises(ModelError, match=name.replace(".", r"\.")):
        load_model(model_dir)


def test_all_missing_required_files_are_listed(model_dir: Path) -> None:
    for name in ("tokenizer.json", "embedding.safetensors"):
        (model_dir / name).unlink()
    with pytest.raises(ModelError) as excinfo:
        load_model(model_dir)
    assert "tokenizer.json" in str(excinfo.value) and "embedding.safetensors" in str(excinfo.value)


def test_missing_vocab_map_for_reduced_vocab(tmp_path: Path) -> None:
    root = make_model(tmp_path / "m", reduced=64)
    (root / "vocab_map.safetensors").unlink()
    with pytest.raises(ModelError, match=r"reduced_vocab_size.*vocab_map\.safetensors"):
        load_model(root)


def test_path_must_be_an_existing_directory(model_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="does not exist"):
        load_model(tmp_path / "nope")
    with pytest.raises(ModelError, match="not a directory"):
        load_model(model_dir / "config.json")


def test_export_root_hints_at_llm_subfolder(tmp_path: Path) -> None:
    make_model(tmp_path / "out" / "llm")
    with pytest.raises(ModelError, match=r"no config\.json.*did you mean"):
        load_model(tmp_path / "out")


@pytest.mark.parametrize("text", ["{not json", "[1, 2]", "", "\xff"])
def test_config_must_be_a_json_object(model_dir: Path, text: str) -> None:
    (model_dir / "config.json").write_bytes(text.encode("latin-1"))
    with pytest.raises(ModelError, match=r"config\.json"):
        load_model(model_dir)


@pytest.mark.parametrize(
    ("overrides", "pattern"),
    [
        ({"engine_role": "visual"}, "engine_role"),
        ({"engine_role": "draft", "spec_decode_type": DROP}, "engine_role"),
        ({"spec_decode_type": "eagle3"}, "speculative decoding"),
        ({"spec_decode_type": "mtp", "engine_role": "base"}, "spec_decode_type 'mtp'"),
    ],
)
def test_unsupported_engine_kinds(tmp_path: Path, overrides: dict, pattern: str) -> None:
    with pytest.raises(ModelError, match=pattern):
        load_model(make_model(tmp_path / "m", config=overrides))


def test_engine_role_and_spec_decode_type_are_optional(tmp_path: Path) -> None:
    info = load_model(make_model(tmp_path / "m", config={"engine_role": DROP, "spec_decode_type": DROP}))
    assert info.cost_model == "dense_transformer"


@pytest.mark.parametrize("text", ["{oops", "[]"])
def test_invalid_generation_config(tmp_path: Path, text: str) -> None:
    with pytest.raises(ModelError, match=r"generation_config\.json"):
        load_model(make_model(tmp_path / "m", generation=text))


@pytest.mark.parametrize(
    ("overrides", "pattern"),
    [
        ({"num_hidden_layers": DROP}, "num_hidden_layers"),
        ({"hidden_size": 0}, "hidden_size"),
        ({"num_attention_heads": "32"}, "num_attention_heads"),
        ({"vocab_size": True}, "vocab_size"),
        ({"head_dim": DROP, "hidden_size": 66}, "head_dim"),
        ({"model": DROP}, "'model'"),
        ({"kv_cache_dtype": "fp4"}, "kv_cache_dtype"),
        ({"eos_token_id": "</s>"}, "eos_token_id"),
        ({"reduced_vocab_size": "64"}, "reduced_vocab_size"),
        ({"external_weight_files": {"file": FFN_FILE}}, "external_weight_files"),
        ({"external_weight_files": [{"kind": "int4_ffn_weights"}]}, r"external_weight_files\[0\]"),
        ({"external_weight_files": [{"file": "../escape.safetensors"}]}, "inside the model folder"),
        # The builder rejects bare names ("Malformed external weight file entry"), so they fail here too.
        ({"external_weight_files": [FFN_FILE]}, r"external_weight_files\[0\] must be an object"),
        ({"external_weight_files": [{"file": 7}]}, r"external_weight_files\[0\] must be an object"),
    ],
)
def test_malformed_config_fields(tmp_path: Path, overrides: dict, pattern: str) -> None:
    with pytest.raises(ModelError, match=pattern):
        load_model(make_model(tmp_path / "m", config=overrides))


def test_reduced_vocabulary_cannot_exceed_the_vocabulary(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match=r"reduced_vocab_size \(257\) is larger than vocab_size \(256\)"):
        load_model(make_model(tmp_path / "m", reduced=VOCAB + 1))
    assert load_model(make_model(tmp_path / "n", reduced=VOCAB)).output_vocab_size == VOCAB


@pytest.mark.parametrize(
    ("kwargs", "name", "constant"),
    [
        ({"config": {"rope_theta": float("nan")}}, "config.json", "NaN"),
        ({"config": {"rms_norm_eps": float("-inf")}}, "config.json", "-Infinity"),
        ({"generation": {"temperature": float("inf")}}, "generation_config.json", "Infinity"),
    ],
)
def test_non_finite_json_numbers_are_invalid(tmp_path: Path, kwargs: dict, name: str, constant: str) -> None:
    # json.loads accepts them, but they are not JSON: the runtime's parser rejects config.json and as_dict()
    # (generation_defaults) would no longer serialise to valid JSON.
    with pytest.raises(ModelError, match=rf"{name.replace('.', r'[.]')} is not valid JSON \({constant} "):
        load_model(make_model(tmp_path / "m", **kwargs))


def test_truncated_external_data(model_dir: Path) -> None:
    data = model_dir / "model.onnx.data"
    os.truncate(data, size(data) // 2)
    with pytest.raises(ModelError, match="truncated"):
        load_model(model_dir)


def test_external_data_outside_the_folder(tmp_path: Path) -> None:
    root = make_model(tmp_path / "m", external_data=False)
    model = onnx.load(str(root / "model.onnx"))
    tensor = next(t for t in model.graph.initializer if t.name == LM_HEAD_INPUT)
    tensor.ClearField("raw_data")
    tensor.data_location = TensorProto.EXTERNAL
    tensor.external_data.add(key="location", value="../shared/weights.bin")
    (root / "model.onnx").write_bytes(model.SerializeToString())
    with pytest.raises(ModelError, match="not a file inside the model folder"):
        load_model(root)


@pytest.mark.parametrize("content", [b"\xff\xfe garbage \x00\x01" * 50, b""])
def test_unreadable_onnx(model_dir: Path, content: bytes) -> None:
    (model_dir / "model.onnx").write_bytes(content)
    with pytest.raises(ModelError, match=r"model\.onnx"):
        load_model(model_dir)


# --------------------------------------------------------------------------- cost model

def test_cost_formulas(model_dir: Path) -> None:
    info = load_model(model_dir)
    linear = ATTN_PARAMS + FFN_PARAMS + VOCAB * HIDDEN
    lm = VOCAB * HIDDEN
    streamed = info.streamed_weight_bytes

    assert info.decode_flops(1024) == 2 * linear + ATTN_FLOPS_PER_PAIR * 1024
    assert info.decode_flops(0) == 2 * linear
    assert info.prefill_flops(5, 3) == 2 * (linear - lm) * 5 + 2 * lm + ATTN_FLOPS_PER_PAIR * (5 * 3 + 5 * 6 // 2)
    assert info.prefill_flops(7) == 2 * (linear - lm) * 7 + 2 * lm + ATTN_FLOPS_PER_PAIR * 28
    # A one-token prefill after n cached tokens costs exactly one decode step over n + 1 keys.
    assert info.prefill_flops(1) == info.decode_flops(1) == 2 * linear + ATTN_FLOPS_PER_PAIR
    assert info.prefill_flops(1, 9) == info.decode_flops(10)
    assert info.prefill_bytes(1, 9) == info.decode_bytes(10)
    assert info.decode_bytes(10) == streamed + KV_BYTES * 10 + 4 * VOCAB
    assert info.prefill_bytes(5, 3) == streamed + KV_BYTES * (3 + 5) + 4 * VOCAB
    assert info.prefill_bytes(5) == streamed + KV_BYTES * 5 + 4 * VOCAB


@pytest.mark.parametrize(
    ("method", "args", "error"),
    [
        ("decode_flops", (-1,), ValueError),
        ("decode_bytes", (-5,), ValueError),
        ("prefill_flops", (1, -1), ValueError),
        ("prefill_bytes", (-1, 0), ValueError),
        ("decode_flops", (1.5,), TypeError),
        ("prefill_flops", ("3",), TypeError),
    ],
)
def test_cost_methods_reject_bad_counts(model_dir: Path, method: str, args: tuple, error: type) -> None:
    with pytest.raises(error):
        getattr(load_model(model_dir), method)(*args)


@pytest.mark.parametrize(
    ("overrides", "mentions"),
    [
        pytest.param({"num_experts": 8, "num_experts_per_tok": 2}, "num_experts", id="experts"),
        pytest.param({"moe": {"n_routed_experts": 64}}, "n_routed_experts", id="nested-experts"),
        pytest.param({"layer_types": ["attention", "mamba"]}, "mamba", id="mamba-layer-type"),
        pytest.param({"ssm_state_size": 128}, "ssm_state_size", id="ssm-key"),
        pytest.param({"block_type": "SSMMixer"}, "SSMMixer", id="ssm-camel-case-value"),
        pytest.param(
            {"num_linear_attn_layers": 1, "num_attention_layers": 1}, "num_linear_attn_layers", id="edgellm-hybrid"
        ),
        pytest.param({"layer_types": ["linear_attention", "full_attention"]}, "linear_attention", id="linear-attn"),
        pytest.param({"architectures": ["Qwen3NextGatedDeltaNet"]}, "DeltaNet", id="deltanet"),
        pytest.param({"num_attention_layers": 1}, "num_attention_layers", id="fewer-attention-layers"),
        pytest.param({"num_attention_layers": 0}, "num_attention_layers is 0", id="no-attention-layers"),
        # The exporter writes sliding_window_size only when the window is active.
        pytest.param({"sliding_window_size": 512}, "sliding_window_size is 512", id="sliding-window"),
        pytest.param(
            {"sliding_window_size": 512, "max_position_embeddings": 4096}, "sliding-window", id="sliding-window-binds"
        ),
        pytest.param({"model": "qwen3_moe"}, "qwen3_moe", id="qwen3_moe"),
        pytest.param({"model": "qwen3_5"}, "qwen3_5", id="qwen3_5"),
        pytest.param({"model": "qwen3_5_moe"}, "qwen3_5_moe", id="qwen3_5_moe"),
        pytest.param({"model": "nemotron_h"}, "nemotron_h", id="nemotron_h"),
        pytest.param({"model": "gemma4"}, "gemma4", id="gemma4"),
        pytest.param({"model": "gemma4_unified"}, "gemma4_unified", id="gemma4_unified"),
        pytest.param({"model": "qwen3_5_text"}, "qwen3_5_text", id="qwen3_5_text"),
        pytest.param({"model": "gemma4_unified_text"}, "gemma4_unified_text", id="gemma4_unified_text"),
    ],
)
def test_cost_model_unavailable_from_config(tmp_path: Path, overrides: dict, mentions: str) -> None:
    info = load_model(make_model(tmp_path / "m", config=overrides))
    assert info.cost_model == "unavailable"
    assert mentions in (info.cost_model_reason or "")
    assert info.decode_flops(10) is None
    assert info.prefill_flops(10, 2) is None
    assert info.decode_bytes(10) is None
    assert info.prefill_bytes(10, 2) is None
    assert info.linear_params == ATTN_PARAMS + FFN_PARAMS + VOCAB * HIDDEN  # facts are still reported
    assert info.as_dict()["cost_model_reason"] == info.cost_model_reason


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"num_experts": 0}, id="zero-experts"),
        pytest.param({"num_linear_attn_layers": 0, "recurrent_state_size": None}, id="hybrid-keys-off"),
        pytest.param({"mamba_config": {}}, id="empty-mamba-config"),
        pytest.param({"note": "classmates discuss assumptions"}, id="ssm-inside-words"),
        pytest.param({"layer_types": ["full_attention", "sliding_attention"]}, id="attention-layer-types"),
        pytest.param({"num_attention_layers": LAYERS}, id="all-layers-attend"),
        pytest.param({"sliding_window_size": -1}, id="sliding-window-off"),
        pytest.param({"sliding_window_size": 0}, id="sliding-window-zero"),
        # A window at least as long as the longest context never limits attention.
        pytest.param({"sliding_window_size": 4096, "max_position_embeddings": 4096}, id="window-never-binds"),
    ],
)
def test_cost_model_stays_dense(tmp_path: Path, overrides: dict) -> None:
    info = load_model(make_model(tmp_path / "m", config=overrides))
    assert info.cost_model == "dense_transformer", info.cost_model_reason


@pytest.mark.parametrize(
    ("kwargs", "mentions"),
    [
        pytest.param(
            {"extra_nodes": (helper.make_node("Int4MoePlugin", ["inputs_embeds"], ["moe"], domain="trt_edgellm"),)},
            "Int4MoePlugin",
            id="moe-plugin",
        ),
        # Real Edge-LLM op name whose camel case hides the word ("NvFP4 Mo E Plugin ...").
        pytest.param(
            _plugin_node("NvFP4MoEPluginGeforce"),
            "mixture-of-experts layers (NvFP4MoEPluginGeforce)",
            id="moe-plugin-camel-case",
        ),
        pytest.param(
            {"extra_nodes": (helper.make_node("update_ssm_state", ["inputs_embeds"], ["ssm"], domain="trt_edgellm"),)},
            "update_ssm_state",
            id="ssm-plugin",
        ),
        pytest.param(
            _plugin_node("QsaAttentionPlugin"), "sparse attention (QsaAttentionPlugin)", id="sparse-attention"
        ),
        pytest.param(
            {"attention": LAYERS, "attention_attrs": {"sliding_window_size": 64}},
            "attention sliding_window_size is 64",
            id="sliding-window-attention",
        ),
        pytest.param(
            {
                "extra_nodes": (
                    helper.make_node(
                        "Int4GroupwiseGemmPluginV2", ["inputs_embeds"], ["broken"], domain="trt_edgellm", gemm_k=64
                    ),
                )
            },
            "gemm_n",
            id="int4-without-gemm-n",
        ),
        pytest.param({"lm_rows": 200}, "lm_head", id="no-lm-head"),
        pytest.param({"attention": 1}, "1 attention node(s) (AttentionPlugin)", id="attention-count"),
        pytest.param(
            {"attention": LAYERS, "attention_attrs": {"num_kv_heads": 4}}, "num_kv_heads", id="attention-heads"
        ),
    ],
)
def test_cost_model_unavailable_from_graph(tmp_path: Path, kwargs: dict, mentions: str) -> None:
    info = load_model(make_model(tmp_path / "m", **kwargs))
    assert info.cost_model == "unavailable"
    assert mentions in (info.cost_model_reason or "")
    assert info.decode_flops(1) is None


def test_unidentified_lm_head_is_still_counted_as_a_linear_layer(tmp_path: Path) -> None:
    info = load_model(make_model(tmp_path / "m", lm_rows=200))
    assert info.lm_head_params == 0
    assert info.linear_params == ATTN_PARAMS + FFN_PARAMS + 200 * HIDDEN


@pytest.mark.parametrize(
    ("attrs", "config"),
    [
        pytest.param({}, None, id="plain"),
        pytest.param({"sliding_window_size": -1}, None, id="no-window"),  # what the real exports write
        pytest.param({"sliding_window_size": 64}, {"max_position_embeddings": 64}, id="window-never-binds"),
    ],
)
def test_matching_attention_plugins_keep_the_cost_model(tmp_path: Path, attrs: dict, config: dict | None) -> None:
    info = load_model(make_model(tmp_path / "m", attention=LAYERS, attention_attrs=attrs, config=config))
    assert info.cost_model == "dense_transformer", info.cost_model_reason


# --------------------------------------------------------------------------- fingerprint

def _random_bytes(n: int, seed: int = 0) -> bytes:
    return random.Random(seed).randbytes(n)


def _flip(path: Path, offset: int) -> None:
    with path.open("r+b") as handle:
        handle.seek(offset)
        byte = handle.read(1)
        handle.seek(offset)
        handle.write(bytes([byte[0] ^ 0xFF]))


BIG = 3 * MIB + 123
SAMPLE = 64 * 1024


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "tree"
    (root / "sub").mkdir(parents=True)
    (root / "config.json").write_text('{"model": "llama"}')
    (root / "sub" / "nested.txt").write_text("nested")
    (root / "big.bin").write_bytes(_random_bytes(BIG))
    return root


def test_fingerprint_is_location_independent(tree: Path, tmp_path: Path) -> None:
    digest = fingerprint(tree)
    assert len(digest) == 64 and int(digest, 16) >= 0
    copy = tmp_path / "elsewhere" / "other-name"
    shutil.copytree(tree, copy)
    assert fingerprint(copy) == digest
    assert fingerprint(tree) == digest


def test_fingerprint_ignores_creation_order_and_mtimes(tree: Path, tmp_path: Path) -> None:
    other = tmp_path / "other"
    (other / "sub").mkdir(parents=True)
    (other / "big.bin").write_bytes((tree / "big.bin").read_bytes())
    (other / "sub" / "nested.txt").write_text("nested")
    (other / "config.json").write_text('{"model": "llama"}')
    os.utime(other / "big.bin", (1_000_000, 1_000_000))
    assert fingerprint(other) == fingerprint(tree)


@pytest.mark.parametrize(
    "offset",
    [0, SAMPLE - 1, (BIG - SAMPLE) // 2, (BIG - SAMPLE) // 2 + SAMPLE - 1, BIG - SAMPLE, BIG - 1],
    ids=["start", "start-last", "middle", "middle-last", "end-first", "end"],
)
def test_fingerprint_sees_changes_in_sampled_regions(tree: Path, offset: int) -> None:
    before = fingerprint(tree)
    _flip(tree / "big.bin", offset)
    assert fingerprint(tree) != before


def test_fingerprint_samples_large_files(tree: Path) -> None:
    before = fingerprint(tree)
    _flip(tree / "big.bin", SAMPLE + 10)  # between the start and middle samples: not read
    assert fingerprint(tree) == before


def test_fingerprint_sees_size_changes(tree: Path) -> None:
    before = fingerprint(tree)
    with (tree / "big.bin").open("ab") as handle:
        handle.write(b"\x00")
    assert fingerprint(tree) != before


@pytest.mark.parametrize(("file_size", "changes"), [(MIB, True), (MIB + 1, False)], ids=["1MiB-full", "1MiB+1-sampled"])
def test_fingerprint_full_hash_boundary(tmp_path: Path, file_size: int, changes: bool) -> None:
    root = tmp_path / "m"
    root.mkdir()
    (root / "weights.bin").write_bytes(_random_bytes(file_size, seed=1))
    before = fingerprint(root)
    _flip(root / "weights.bin", 200_000)  # outside all three 64 KiB samples
    assert (fingerprint(root) != before) is changes


def test_fingerprint_sees_small_file_edits_renames_and_additions(tree: Path) -> None:
    digests = {fingerprint(tree)}
    (tree / "config.json").write_text('{"model": "qwen3"}')  # same size, different content
    digests.add(fingerprint(tree))
    (tree / "sub" / "nested.txt").rename(tree / "sub" / "renamed.txt")
    digests.add(fingerprint(tree))
    (tree / "sub" / "renamed.txt").rename(tree / "renamed.txt")
    digests.add(fingerprint(tree))
    (tree / "empty.txt").touch()
    digests.add(fingerprint(tree))
    assert len(digests) == 5


def test_fingerprint_skips_broken_symlinks(tree: Path) -> None:
    before = fingerprint(tree)
    (tree / "dangling").symlink_to(tree / "missing-target")
    assert fingerprint(tree) == before


def test_fingerprint_requires_a_directory(tree: Path) -> None:
    with pytest.raises(ModelError):
        fingerprint(tree / "config.json")
    with pytest.raises(ModelError):
        fingerprint(tree / "missing")


needs_permissions = pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root ignores directory permissions"
)


@needs_permissions
def test_unlistable_folder_is_an_error(model_dir: Path) -> None:
    # Files can still be opened by name (x without r), but an empty listing would give every such folder the
    # same fingerprint, so a changed model would silently reuse a stale engine.
    model_dir.chmod(0o300)
    try:
        with pytest.raises(ModelError, match="cannot list the files in"):
            fingerprint(model_dir)
        with pytest.raises(ModelError, match="cannot list the files in"):
            load_model(model_dir)
    finally:
        model_dir.chmod(0o755)


@needs_permissions
def test_unlistable_subfolder_is_an_error(tree: Path) -> None:
    (tree / "sub").chmod(0o100)  # its files can be opened by name (config may point there) but not listed
    try:
        with pytest.raises(ModelError, match=r"cannot list the files in .*sub"):
            fingerprint(tree)
    finally:
        (tree / "sub").chmod(0o755)


@needs_permissions
def test_untraversable_parent_folder_is_a_model_error(model_dir: Path) -> None:
    # pathlib raises PermissionError (not False) for EACCES; callers expect a ModelError with the path.
    model_dir.parent.chmod(0o600)
    try:
        with pytest.raises(ModelError, match=r"cannot read the model folder .*Permission denied|does not exist"):
            load_model(model_dir)
        with pytest.raises(ModelError):
            fingerprint(model_dir)
    finally:
        model_dir.parent.chmod(0o755)


def test_fingerprint_of_a_model_folder(model_dir: Path, tmp_path: Path) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(model_dir, copy)
    assert fingerprint(copy) == fingerprint(model_dir)
    (copy / "chat_template.jinja").write_text("{{ other }}")
    assert fingerprint(copy) != fingerprint(model_dir)


# --------------------------------------------------------------------------- estimate_memory

@pytest.mark.parametrize(("kv_capacity", "pages"), [(0, 0), (1, 1), (128, 1), (129, 2), (3584, 28)])
def test_estimate_memory_arithmetic(model_dir: Path, kv_capacity: int, pages: int) -> None:
    info = replace(load_model(model_dir), resident_weight_bytes=5_000_000, kv_bytes_per_token=1_000, hidden_size=64)
    estimate = estimate_memory(info, max_batch_size=3, kv_capacity=kv_capacity, max_input_len=100)
    assert list(estimate) == ["weights", "kv_cache", "activations", "runtime", "total"]
    assert estimate["weights"] == 5_000_000
    assert estimate["kv_cache"] == 3 * pages * 128 * 1_000
    assert estimate["activations"] == 3 * 100 * 64 * 2 * 20
    assert estimate["runtime"] == 300 * MIB
    assert estimate["total"] == sum(estimate[key] for key in ("weights", "kv_cache", "activations", "runtime"))


@pytest.mark.parametrize(
    ("args", "error"), [((-1, 128, 10), ValueError), ((1, -1, 10), ValueError), ((1, 128, 2.5), TypeError)]
)
def test_estimate_memory_rejects_bad_arguments(model_dir: Path, args: tuple, error: type) -> None:
    with pytest.raises(error):
        estimate_memory(load_model(model_dir), *args)


# --------------------------------------------------------------------------- real exports (skipped when absent)

needs_full = pytest.mark.skipif(not (FULL / "config.json").is_file(), reason="models/full_vocab_model not present")
needs_reduced = pytest.mark.skipif(
    not (REDUCED / "config.json").is_file(), reason="models/reduced_vocab_model not present"
)
REAL_ATTENTION_PARAMS = 1_342_177_280
REAL_FFN_PARAMS = 3_623_878_656  # pruned widths summing to 294,912 over 32 layers


def _check_real_common(info: ModelInfo) -> None:
    assert info.model_type == "llama"
    assert info.edgellm_version == "0.11.0"
    shape = (info.num_layers, info.hidden_size, info.num_heads, info.num_kv_heads, info.head_dim)
    assert shape == (32, 4096, 32, 8, 128)
    assert info.vocab_size == 128_256
    assert info.kv_cache_dtype == "fp16"
    assert info.kv_bytes_per_token == 131_072
    assert info.eos_token_ids == (128_001, 128_008, 128_009)
    assert info.quantization == "INT4 groupwise W4A16 (group 128), fp16 lm_head"
    assert info.linear_params - info.lm_head_params == REAL_ATTENTION_PARAMS + REAL_FFN_PARAMS
    assert info.embedding_params == 525_336_576
    assert info.has_chat_template is True
    assert info.cost_model == "dense_transformer" and info.cost_model_reason is None
    assert json.loads(json.dumps(info.as_dict()))["name"] == info.name


@needs_full
def test_real_full_vocab_model() -> None:
    info = load_model(FULL)
    _check_real_common(info)
    assert info.name == "full_vocab_model"
    assert info.output_vocab_size == 128_256
    assert info.lm_head_params == 525_336_576
    assert info.linear_params == 5_491_392_512
    assert info.streamed_weight_bytes == 3_620_501_672
    assert info.resident_weight_bytes == 3_620_501_672 + info.file_bytes["embedding.safetensors"]
    assert info.decode_flops(1024) == 11_519_655_936
    assert info.decode_flops(1024) == 2 * 5_491_392_512 + 4 * 32 * 32 * 128 * 1024


@needs_reduced
def test_real_reduced_vocab_model() -> None:
    info = load_model(REDUCED)
    _check_real_common(info)
    assert info.output_vocab_size == 16_384
    assert info.lm_head_params == 67_108_864
    assert info.linear_params == 5_033_164_800
    assert info.streamed_weight_bytes == 2_704_046_248
    assert info.resident_weight_bytes == (
        2_704_046_248 + info.file_bytes["embedding.safetensors"] + info.file_bytes["vocab_map.safetensors"]
    )
    estimate = estimate_memory(info, max_batch_size=1, kv_capacity=3584, max_input_len=3072)
    assert estimate["kv_cache"] == 28 * 128 * 131_072


@needs_full
@needs_reduced
def test_real_fingerprints_are_stable_and_distinct() -> None:
    full = fingerprint(FULL)
    assert fingerprint(FULL) == full
    assert fingerprint(REDUCED) != full


def test_fingerprint_ignores_files_the_engine_build_never_reads(tree: Path) -> None:
    before = fingerprint(tree)
    (tree / "generation_config.json").write_text('{"temperature": 0.7}')
    (tree / "README.md").write_text("notes")
    (tree / "LICENSE.txt").write_text("licence")
    (tree / ".cache" / "huggingface").mkdir(parents=True)  # left by `hf download --local-dir`
    (tree / ".cache" / "huggingface" / "generation_config.json.metadata").write_text("etag")
    (tree / ".DS_Store").write_bytes(b"\0")
    assert fingerprint(tree) == before
    (tree / "generation_config.json").write_text('{"temperature": 0.1, "max_new_tokens": 64}')
    assert fingerprint(tree) == before
    (tree / "chat_template.jinja").write_text("{{ changed }}")  # copied into the engine folder: must count
    assert fingerprint(tree) != before


def test_weights_count_only_the_external_data_the_graph_uses(tmp_path: Path) -> None:
    from conftest import make_tiny_model

    folder = make_tiny_model(tmp_path)
    before = load_model(folder)
    model = onnx.load(str(folder / "model.onnx"))
    onnx.save_model(model, str(folder / "model.onnx"), save_as_external_data=True, all_tensors_to_one_file=True,
                    location="model.onnx.data", size_threshold=0)
    with open(folder / "model.onnx.data", "ab") as data:  # bytes an exporter left behind after moving weights out
        data.write(b"\0" * 1_000_000)
    after = load_model(folder)
    used = sum(int({e.key: e.value for e in t.external_data}["length"]) for t in model.graph.initializer)
    assert after.streamed_weight_bytes == used < (folder / "model.onnx.data").stat().st_size - 999_999
    assert after.quantization == before.quantization and after.linear_params == before.linear_params


def test_zero_point_corrections_belong_to_their_int4_layer(tmp_path: Path) -> None:
    import numpy as np
    from onnx import helper, numpy_helper

    from conftest import make_tiny_model

    folder = make_tiny_model(tmp_path)
    before = load_model(folder)
    model = onnx.load(str(folder / "model.onnx"))
    graph = model.graph
    corr = numpy_helper.from_array(np.zeros((1, 64), dtype=np.float16), "_model.layers.0.self_attn.q_proj.zp_correction")
    graph.initializer.append(corr)
    graph.node.insert(0, helper.make_node("MatMul", ["means", corr.name], ["correction_out"]))
    graph.input.append(helper.make_tensor_value_info("means", onnx.TensorProto.FLOAT16, ["tokens", 1]))
    onnx.save(model, str(folder / "model.onnx"))
    after = load_model(folder)
    assert after.quantization == before.quantization  # no "+ FP16": the correction is part of the INT4 layer
    assert after.linear_params == before.linear_params + 64  # but its work is counted
