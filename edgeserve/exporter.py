"""``edgeserve export``: turn a Hugging Face checkpoint into a model folder edgeserve can serve.

The checkpoint is a Hugging Face model ID or a local folder: an original model, a fine-tuned or
pruned one, or one quantized to INT4 with AWQ or GPTQ. TensorRT Edge-LLM's exporter does the work.
It is installed on first use into ``~/.local/share/edgeserve/exporter/`` (its own virtual
environment, about 5 GB with PyTorch), at the Edge-LLM commit edgeserve's GPU runtime is built
from, with edgeserve's exporter patches (``edgeserve/export_patches``: per-layer MLP widths for
pruned checkpoints). Exporting runs on the CPU and needs no GPU or ``edgeserve setup``.

The result is one folder (Edge-LLM's ``llm`` export, plus the checkpoint's generation_config.json)
that ``edgeserve serve --model`` takes. Visual and audio parts of multimodal checkpoints are left
out: edgeserve serves text.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.resources
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from edgeserve import __version__, config, nativebuild

EDGELLM_REPO = "https://github.com/NVIDIA/TensorRT-Edge-LLM.git"
# The exporter's wheel ships this submodule's licence; the C++ submodules are not needed to export.
_SUBMODULES = ("3rdParty/nlohmannJson",)
_INT4_METHODS = ("awq", "gptq")


class ExportError(RuntimeError):
    """The export failed; the message says why and what to do."""


def exporter_root() -> Path:
    return config.data_dir() / "exporter"


def patches_dir() -> Path:
    return Path(str(importlib.resources.files("edgeserve") / "export_patches"))


def patch_files() -> list[Path]:
    return sorted(patches_dir().glob("*.patch"))


def environment_key() -> str:
    """Identifies an exporter installation: the Edge-LLM commit and edgeserve's exporter patches."""
    digest = hashlib.sha256(nativebuild.edgellm_commit().encode())
    for patch in patch_files():
        digest.update(patch.name.encode() + b"\0" + patch.read_bytes())
    return digest.hexdigest()[:16]


# ----------------------------------------------------------------------------------------------- environment

@dataclass(frozen=True)
class Exporter:
    root: Path
    python: Path
    export: Path  # tensorrt-edgellm-export


def ensure(progress: Callable[[str], None] = print) -> Exporter:
    """The exporter environment, installed (or brought up to date) first if needed."""
    root = exporter_root()
    root.mkdir(parents=True, exist_ok=True)
    venv = root / "venv"
    found = Exporter(root=root, python=venv / "bin" / "python", export=venv / "bin" / "tensorrt-edgellm-export")
    key = environment_key()
    with nativebuild.file_lock(root / ".lock", progress, "another edgeserve export is installing the exporter"):
        if _installed_key(root) == key and found.export.exists():
            return found
        progress(f"installing TensorRT Edge-LLM's exporter in {root} (once: a few minutes, about 5 GB)")
        log_path = root / "install.log"
        env = dict(os.environ)
        env.pop("EDGESERVE_API_KEY", None)
        src = root / "src"
        commit = nativebuild.edgellm_commit()
        if not (src / ".git").is_dir():  # an interrupted clone is started again
            partial = root / "src.partial"
            shutil.rmtree(src, ignore_errors=True)
            shutil.rmtree(partial, ignore_errors=True)
            _step(["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", EDGELLM_REPO, str(partial)], env,
                  log_path)
            os.replace(partial, src)
        if _git(src, "cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
            _step(["git", "-C", str(src), "fetch", "--quiet", "origin"], env, log_path)
        _step(["git", "-C", str(src), "checkout", "--quiet", "--force", commit], env, log_path)
        _step(["git", "-C", str(src), "clean", "--quiet", "-fdx", "--exclude=3rdParty"], env, log_path)
        _step(["git", "-C", str(src), "submodule", "update", "--quiet", "--init", *_SUBMODULES], env, log_path)
        for patch in patch_files():
            _step(["git", "-C", str(src), "apply", str(patch)], env, log_path)
            progress(f"applied {patch.name}")
        shutil.rmtree(venv, ignore_errors=True)
        _step([sys.executable, "-m", "venv", str(venv)], env, log_path)
        _step([str(venv / "bin" / "pip"), "install", "--quiet", "--upgrade", "pip"], env, log_path)
        progress("installing the exporter's Python packages: PyTorch, Transformers, ONNX (a few minutes, quietly)")
        _step([str(venv / "bin" / "pip"), "install", "--quiet", f"{src}[export]"], env, log_path)
        (root / "installed.json").write_text(json.dumps({
            "key": key, "edgellm_commit": commit, "patches": [p.name for p in patch_files()],
            "edgeserve_version": __version__,
            "installed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        }, indent=2) + "\n")
    return found


def _installed_key(root: Path) -> str | None:
    try:
        return json.loads((root / "installed.json").read_text()).get("key")
    except (OSError, ValueError, AttributeError):
        return None


def _git(src: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(src), *args], capture_output=True, text=True)


def _step(command: list[str], env: dict[str, str], log_path: Path) -> None:
    if nativebuild.run_logged(command, env, log_path, show=False) != 0:
        raise ExportError(f"installing the exporter failed at `{' '.join(command[:4])} …`: the last lines are in "
                          f"{log_path}; check the internet connection and run the command again")


# ----------------------------------------------------------------------------------------------- export

def export(checkpoint: str, model_dir: Path, extra: list[str] | None = None,
           progress: Callable[[str], None] = print) -> Path:
    """Export ``checkpoint`` into ``model_dir`` (which must not exist yet, or be empty); returns ``model_dir``."""
    model_dir = Path(model_dir).expanduser().absolute()
    if model_dir.exists() and (not model_dir.is_dir() or any(model_dir.iterdir())):
        raise ExportError(f"{model_dir} already exists and is not empty: choose a new folder, or delete it first")
    local = Path(checkpoint).expanduser()
    source = str(local.absolute()) if local.is_dir() else checkpoint
    if local.is_dir() and not (local / "config.json").is_file():
        raise ExportError(f"{local} has no config.json, so it is not a Hugging Face checkpoint folder (see "
                          "docs/EXPORT.md, 'Your own checkpoints')")
    exporter = ensure(progress)
    hf_config = _json(_checkpoint_file(exporter, source, "config.json", required=True))
    options = export_options(hf_config, extra or [])
    for note in describe(hf_config):
        progress(note)

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = model_dir.parent / f".{model_dir.name}.export-{os.getpid()}"
    shutil.rmtree(partial, ignore_errors=True)
    log_dir = config.default_cache_dir() / "exports"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{model_dir.name}.log"
    log_path.unlink(missing_ok=True)
    progress(f"exporting {checkpoint} (log: {log_path}); this takes a few minutes")
    env = dict(os.environ)
    env.pop("EDGESERVE_API_KEY", None)
    try:
        code = nativebuild.run_logged([str(exporter.export), source, str(partial), *options], env, log_path)
        if code != 0:
            raise ExportError(f"the export failed (exit {code}); the exporter's output is above and in {log_path}")
        llm = partial / "llm"
        if not (llm / "config.json").is_file():
            raise ExportError(f"the exporter wrote no language model (no {llm / 'config.json'}); see {log_path}")
        dropped = sorted(p.name for p in partial.iterdir() if p.is_dir() and p.name != "llm")
        if dropped:
            progress(f"left out the {', '.join(dropped)} part(s): edgeserve serves the language model only")
        if model_dir.exists():
            model_dir.rmdir()  # empty, checked above
        os.replace(llm, model_dir)
    finally:
        shutil.rmtree(partial, ignore_errors=True)
    saved = compact_onnx_data(model_dir)
    if saved:
        progress(f"removed {saved / 2**20:.0f} MB of unused bytes from model.onnx.data")

    generation = _checkpoint_file(exporter, source, "generation_config.json", required=False)
    if generation is not None:
        _json(generation)  # a broken file is reported now, not when the server starts
        shutil.copyfile(generation, model_dir / "generation_config.json")
    else:
        progress("the checkpoint has no generation_config.json: requests without sampling settings decode greedily")
    return model_dir


def export_options(hf_config: dict, extra: list[str]) -> list[str]:
    """Exporter options for this checkpoint, unless the caller chose --externalize-weights itself.

    The weights that dominate the engine build's memory are kept outside the engine, where the runtime loads
    them as they are: INT4 feed-forward weights (as Edge-LLM recommends for Jetson) and an FP16 output layer
    (lm_head). TensorRT holds the weights it builds with several times, so this is what lets INT4 models with
    large vocabularies build on an 8 GB Jetson; decoding speed is the same.
    """
    if any(option.startswith("--externalize-weights") for option in extra):
        return list(extra)
    kinds = []
    if is_int4(hf_config):
        kinds.append("int4_ffn")
        text = hf_config.get("text_config") if isinstance(hf_config.get("text_config"), dict) else hf_config
        if any(int(text.get(key) or 0) > 0 for key in ("num_experts", "n_routed_experts", "num_local_experts")):
            kinds.append("int4_moe")
    if fp16_lm_head(hf_config):
        kinds.append("lm_head")
    return ["--externalize-weights", *kinds, *extra] if kinds else list(extra)


def fp16_lm_head(hf_config: dict) -> bool:
    """Whether the checkpoint's output layer is unquantized: an FP16/BF16 checkpoint, or INT4 AWQ / GPTQ that left
    lm_head alone (Edge-LLM can keep only an unquantized lm_head outside the engine)."""
    quant = hf_config.get("quantization_config")
    if not isinstance(quant, dict):
        return True
    if not is_int4(hf_config):
        return False  # FP8, NVFP4, ...: the lm_head may be quantized
    return not bool(quant.get("lm_head", False))


def is_int4(hf_config: dict) -> bool:
    quant = hf_config.get("quantization_config")
    if not isinstance(quant, dict):
        return False
    return str(quant.get("quant_method", "")).lower() in _INT4_METHODS and int(quant.get("bits", 4) or 4) == 4


def describe(hf_config: dict) -> list[str]:
    """What the export will treat specially, in words."""
    text = hf_config.get("text_config") if isinstance(hf_config.get("text_config"), dict) else hf_config
    notes = []
    if is_int4(hf_config):
        quant = hf_config["quantization_config"]
        zeros = "symmetric" if quant.get("sym") else "asymmetric zero points, kept exact"
        notes.append(f"INT4 {quant.get('quant_method', '').upper()} checkpoint (group size "
                     f"{quant.get('group_size', '?')}, {zeros}): its feed-forward weights are kept outside the engine")
    widths = text.get("intermediate_sizes")
    if isinstance(widths, list) and widths:
        notes.append(f"pruned checkpoint: per-layer MLP widths {min(widths)}-{max(widths)} (intermediate_sizes)")
    return notes


def compact_onnx_data(model_dir: Path) -> int:
    """Rewrite model.onnx.data with only the bytes model.onnx uses; returns the bytes saved (0: nothing to do).

    The exporter moves weights to external weight files (--externalize-weights) but leaves their bytes in
    model.onnx.data, which can then be mostly unused space.
    """
    import onnx
    from onnx.external_data_helper import load_external_data_for_model

    data = model_dir / "model.onnx.data"
    if not data.is_file():
        return 0
    model = onnx.load(str(model_dir / "model.onnx"), load_external_data=False)
    used = sum(int({e.key: e.value for e in t.external_data}.get("length") or 0)
               for t in model.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL)
    size = data.stat().st_size
    if used >= size * 0.9:
        return 0
    load_external_data_for_model(model, str(model_dir))
    staging = model_dir / ".compact"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    try:
        onnx.save_model(model, str(staging / "model.onnx"), save_as_external_data=True, all_tensors_to_one_file=True,
                        location="model.onnx.data", size_threshold=1024)
        os.replace(staging / "model.onnx.data", data)
        os.replace(staging / "model.onnx", model_dir / "model.onnx")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return size - data.stat().st_size


def _checkpoint_file(exporter: Exporter, source: str, name: str, required: bool) -> Path | None:
    """A file of the checkpoint: in its folder, else downloaded from the Hugging Face Hub (with the user's login)."""
    local = Path(source)
    if local.is_dir():
        return local / name if (local / name).is_file() else None
    script = "import sys\nfrom huggingface_hub import hf_hub_download\nprint(hf_hub_download(sys.argv[1], sys.argv[2]))"
    result = subprocess.run([str(exporter.python), "-c", script, source, name], capture_output=True, text=True)
    if result.returncode != 0:
        if not required:
            return None
        reason = (result.stderr.strip().splitlines() or ["unknown error"])[-1]
        raise ExportError(f"cannot read {name} of {source} from Hugging Face ({reason[:300]}); check the model name, "
                          "and for a gated model accept its license on huggingface.co and set HF_TOKEN to your access "
                          "token")
    return Path(result.stdout.strip().splitlines()[-1])


def _json(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ExportError(f"cannot read {path}: {e}") from e
    if not isinstance(data, dict):
        raise ExportError(f"{path} is not a JSON object")
    return data


def run(checkpoint: str, model_dir: Path, extra: list[str]) -> int:
    from edgeserve.model import ModelError, estimate_memory, load_model

    try:
        folder = export(checkpoint, model_dir, extra, progress=_say)
    except ExportError as e:
        print(f"edgeserve export: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("edgeserve export: interrupted", file=sys.stderr)
        return 130
    try:
        model = load_model(folder)
    except ModelError as e:
        print(f"edgeserve export: the exported folder cannot be served: {e}", file=sys.stderr)
        return 1
    need = estimate_memory(model, 1, 3712, 3072)["total"] / 2**30
    _say(f"exported {checkpoint} to {folder}")
    _say(f"  {model.model_type}, {model.num_layers} layers, {model.quantization}; about {need:.1f} GiB to serve with "
         "the default limits")
    print(f"\nNext: edgeserve serve --model {folder}")
    return 0


def _say(text: str) -> None:
    print(f"[edgeserve export] {text}", flush=True)


__all__ = ["ExportError", "Exporter", "compact_onnx_data", "describe", "ensure", "environment_key", "export",
           "export_options", "exporter_root", "fp16_lm_head", "is_int4", "patch_files", "patches_dir", "run"]
