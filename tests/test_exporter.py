"""``edgeserve export`` with a stand-in exporter environment (no download, no PyTorch): options per checkpoint, the
model folder it writes, the sampling defaults it copies, and its errors."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest

from conftest import make_tiny_model
from edgeserve import cli, exporter

SKIPS = ["--skip-visual", "--skip-audio"]  # passed to every export: only the language model is served


@pytest.fixture
def fake_exporter(tmp_path, monkeypatch) -> dict:
    """An exporter environment whose tensorrt-edgellm-export writes a tiny Edge-LLM export and records its arguments
    (FAIL_EXPORT=1 makes it fail; VISUAL=1 adds a visual/ part like a multimodal export)."""
    root = tmp_path / "exporter"
    bin_dir = root / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    calls = tmp_path / "calls.jsonl"
    script = bin_dir / "export_impl.py"
    script.write_text(f"""
import json, os, sys
sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
from conftest import make_tiny_model
with open({str(calls)!r}, "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\\n")
if os.environ.get("FAIL_EXPORT"):
    print("Traceback: something broke", flush=True)
    sys.exit(3)
from pathlib import Path
out = Path(sys.argv[2])
make_tiny_model(out, name="llm")
if os.environ.get("VISUAL"):
    (out / "visual").mkdir()
""")
    export = bin_dir / "tensorrt-edgellm-export"
    export.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(script))} \"$@\"\n")
    export.chmod(0o755)
    found = exporter.Exporter(root=root, python=Path(sys.executable), export=export)
    monkeypatch.setattr(exporter, "ensure", lambda progress=print: found)
    return {"calls": calls, "exporter": found}


def _checkpoint(root: Path, **config) -> Path:
    """A local Hugging Face checkpoint folder (only what export reads itself: config.json, generation_config.json)."""
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"model_type": "llama", "num_hidden_layers": 1, **config}))
    (root / "model.safetensors").write_bytes(b"")
    return root


def _calls(fake) -> list[list[str]]:
    return [json.loads(line) for line in fake["calls"].read_text().splitlines()]


def test_a_local_checkpoint_becomes_a_model_folder_with_its_sampling_defaults(fake_exporter, tmp_path, capsys):
    ckpt = _checkpoint(tmp_path / "my-finetune")
    (ckpt / "generation_config.json").write_text('{"temperature": 0.6, "top_p": 0.9}')
    model_dir = tmp_path / "models" / "my-finetune"
    assert cli.main(["export", str(ckpt), str(model_dir)]) == 0
    assert (model_dir / "config.json").is_file() and (model_dir / "model.onnx").is_file()
    assert json.loads((model_dir / "generation_config.json").read_text()) == {"temperature": 0.6, "top_p": 0.9}
    (call,) = _calls(fake_exporter)
    assert call[0] == str(ckpt) and call[2:] == ["--externalize-weights", "lm_head", *SKIPS]  # FP16: the lm_head
    assert not list(model_dir.parent.glob(".my-finetune.export-*"))  # the staging folder is gone
    out = capsys.readouterr().out
    assert f"Next: edgeserve serve --model {model_dir}" in out and "llama, 1 layers" in out


@pytest.mark.parametrize(("quant", "expected"), [
    ({"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": False},
     ["--externalize-weights", "int4_ffn", "lm_head"]),
    ({"quant_method": "awq", "bits": 4, "group_size": 128}, ["--externalize-weights", "int4_ffn", "lm_head"]),
    ({"quant_method": "gptq", "bits": 4, "lm_head": True}, ["--externalize-weights", "int4_ffn"]),  # quantized
    ({"quant_method": "gptq", "bits": 8}, []),
    ({"quant_method": "fp8"}, []),  # its lm_head may be quantized: left to the exporter
])
def test_the_big_weights_are_kept_outside_the_engine(fake_exporter, tmp_path, quant, expected):
    ckpt = _checkpoint(tmp_path / "ckpt", quantization_config=quant)
    assert cli.main(["export", str(ckpt), str(tmp_path / "out")]) == 0
    assert _calls(fake_exporter)[0][2:] == [*expected, *SKIPS]


@pytest.mark.parametrize(("sidecar", "quant", "expected"), [
    # NVIDIA ModelOpt (e.g. tensorrt-edgellm-quantize --quantization int4_awq): hf_quant_config.json next to config.json
    ({"quantization": {"quant_algo": "W4A16_AWQ", "group_size": 128, "exclude_modules": ["lm_head"]}}, None,
     ["--externalize-weights", "int4_ffn", "lm_head"]),
    ({"quantization": {"quant_algo": "W4A16_AWQ", "group_size": 128}}, None, ["--externalize-weights", "int4_ffn"]),
    ({"quantization": {"quant_algo": "FP8", "exclude_modules": ["lm_head"]}}, None, []),
    # ... or the same inside config.json; the separate file wins, as in Edge-LLM
    (None, {"quant_method": "modelopt", "quant_algo": "W4A16_AWQ", "ignore": ["lm_head"]},
     ["--externalize-weights", "int4_ffn", "lm_head"]),
    ({"quantization": {"quant_algo": "NVFP4"}}, {"quant_method": "awq", "bits": 4}, []),
])
def test_modelopt_int4_checkpoints_keep_their_big_weights_outside_too(fake_exporter, tmp_path, sidecar, quant,
                                                                      expected):
    ckpt = _checkpoint(tmp_path / "ckpt", **({"quantization_config": quant} if quant else {}))
    if sidecar is not None:
        (ckpt / "hf_quant_config.json").write_text(json.dumps(sidecar))
    assert cli.main(["export", str(ckpt), str(tmp_path / "out")]) == 0
    assert _calls(fake_exporter)[0][2:] == [*expected, *SKIPS]


@pytest.mark.parametrize("checkpoint", ["./no-such-folder", "/no/such/folder", "~/no-such-folder", "a/b/c"])
def test_a_missing_checkpoint_folder_is_not_looked_up_on_hugging_face(fake_exporter, tmp_path, checkpoint, capsys):
    assert cli.main(["export", checkpoint, str(tmp_path / "out")]) == 1
    assert "does not exist: pass your checkpoint folder" in capsys.readouterr().err
    assert not fake_exporter["calls"].exists() or not _calls(fake_exporter)


def test_installing_the_exporter_without_git_says_so(monkeypatch):
    monkeypatch.setattr(exporter.shutil, "which", lambda name: None)
    with pytest.raises(exporter.ExportError, match=r"needs git.*sudo apt install git"):
        exporter.ensure(progress=lambda text: None)


def test_moe_and_pruned_checkpoints_are_described(tmp_path):
    hf = {"quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128}, "num_experts": 8,
          "intermediate_sizes": [4352, 14336]}
    assert exporter.export_options(hf, []) == ["--externalize-weights", "int4_ffn", "int4_moe", "lm_head", *SKIPS]
    notes = exporter.describe(hf)
    assert any("INT4 GPTQ" in n for n in notes) and any("per-layer MLP widths 4352-14336" in n for n in notes)


def test_options_after_a_double_dash_reach_the_exporter_unchanged(fake_exporter, tmp_path):
    ckpt = _checkpoint(tmp_path / "ckpt", quantization_config={"quant_method": "awq", "bits": 4})
    assert cli.main(["export", str(ckpt), str(tmp_path / "out"), "--", "--reduced-vocab-dir", "/v",
                     "--externalize-weights", "all"]) == 0
    assert _calls(fake_exporter)[0][2:] == ["--reduced-vocab-dir", "/v", "--externalize-weights", "all", *SKIPS]
    assert exporter.export_options({}, ["--skip-visual"]) == ["--externalize-weights", "lm_head", "--skip-visual",
                                                              "--skip-audio"]  # not given twice


def test_a_multimodal_export_keeps_only_the_language_model(fake_exporter, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VISUAL", "1")
    assert cli.main(["export", str(_checkpoint(tmp_path / "vlm")), str(tmp_path / "out")]) == 0
    assert "left out the visual part" in capsys.readouterr().out
    assert not (tmp_path / "out" / "visual").exists()


def test_a_failed_export_leaves_nothing_behind(fake_exporter, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAIL_EXPORT", "1")
    model_dir = tmp_path / "out"
    assert cli.main(["export", str(_checkpoint(tmp_path / "ckpt")), str(model_dir)]) == 1
    err = capsys.readouterr().err
    assert "the export failed (exit 3)" in err and "exports" in err  # names the log
    assert not model_dir.exists() and not list(tmp_path.glob(".out.export-*"))


def test_export_refuses_an_existing_folder_and_a_folder_that_is_not_a_checkpoint(fake_exporter, tmp_path, capsys):
    existing = make_tiny_model(tmp_path, name="existing")
    assert cli.main(["export", str(_checkpoint(tmp_path / "ckpt")), str(existing)]) == 1
    assert "already exists and is not empty" in capsys.readouterr().err
    (tmp_path / "not-a-checkpoint").mkdir()
    assert cli.main(["export", str(tmp_path / "not-a-checkpoint"), str(tmp_path / "out")]) == 1
    assert "has no config.json" in capsys.readouterr().err
    assert not (fake_exporter["calls"]).exists()  # the exporter never ran


def test_an_empty_target_folder_is_fine(fake_exporter, tmp_path):
    (tmp_path / "out").mkdir()
    assert cli.main(["export", str(_checkpoint(tmp_path / "ckpt")), str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "config.json").is_file()


def test_the_environment_key_follows_the_commit_and_the_patches():
    key = exporter.environment_key()
    assert len(key) == 16 and key == exporter.environment_key()
    names = [p.name for p in exporter.patch_files()]
    assert names == ["0001-per-layer-mlp-widths.patch", "0002-exact-int4-zero-points.patch"]


def test_unused_bytes_are_removed_from_the_onnx_data_file(tmp_path):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    folder = tmp_path / "m"
    folder.mkdir()
    weight = numpy_helper.from_array(np.arange(4096, dtype=np.float32).reshape(64, 64), "w")
    graph = helper.make_graph([helper.make_node("MatMul", ["x", "w"], ["y"])], "g",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 64])], initializer=[weight])
    onnx.save_model(helper.make_model(graph), str(folder / "model.onnx"), save_as_external_data=True,
                    all_tensors_to_one_file=True, location="model.onnx.data", size_threshold=0)
    with open(folder / "model.onnx.data", "ab") as data:  # what an exporter leaves after moving weights out
        data.write(b"\0" * 200_000)
    assert exporter.compact_onnx_data(folder) == 200_000
    assert (folder / "model.onnx.data").stat().st_size == 4096 * 4
    reloaded = onnx.load(str(folder / "model.onnx"))
    assert np.array_equal(numpy_helper.to_array(reloaded.graph.initializer[0]),
                          np.arange(4096, dtype=np.float32).reshape(64, 64))
    assert exporter.compact_onnx_data(folder) == 0  # nothing left to remove
    assert not (folder / ".compact").exists()
