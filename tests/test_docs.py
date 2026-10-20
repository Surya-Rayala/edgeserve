"""The documentation matches the code: every option and setting is documented, every ``edgeserve`` command in the
docs parses, nothing refers to removed files or settings, every referenced section and link exists, versions
match, and the README's API examples run against a real server (with the fake worker)."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import edgeserve
from conftest import REPO, edgeserve_server, make_tiny_model
from edgeserve import batch, cli, config, nativebuild
from edgeserve.service import INT4_PLUGIN_SMS

README = REPO / "README.md"
DOCS = sorted((REPO / "docs").glob("*.md"))
# models/README.md describes one deployment's local model folders and may be kept out of a published repository.
PAGES = [README, *DOCS, *[page for page in [REPO / "models" / "README.md"] if page.is_file()]]
GITHUB = re.compile(r"https://github\.com/[^/]+/edgeserve/blob/main/([^)#\s]+)(#[^)\s]*)?")
pytestmark = pytest.mark.skipif(not README.is_file() or not DOCS, reason="needs the source tree")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _blocks(path: Path, language: str | None = None) -> list[str]:
    return [body for lang, body in re.findall(r"```(\w*)\n(.*?)```", _text(path), re.S)
            if language is None or lang == language]


def _section(path: Path, heading: str) -> str:
    text = _text(path)
    start = text.index(f"\n## {heading}\n")
    end = text.find("\n## ", start + 1)
    return text[start:end if end != -1 else None]


def _headings(path: Path) -> set[str]:
    """Headings, also without a step number ("3. Export a model" is found as "Export a model" too)."""
    found = {m.group(1).strip() for m in re.finditer(r"^#{1,6} (.+)$", _text(path), re.M)}
    return found | {re.sub(r"^\d+\. ", "", heading) for heading in found}


def _anchor(heading: str) -> str:
    """GitHub's anchor for a heading."""
    anchor = re.sub(r"[^\w\- ]", "", heading.lower().replace("`", "")).replace(" ", "-")
    return anchor


def _parsers() -> dict[str, argparse.ArgumentParser]:
    parser = cli._parser()
    subparsers = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    found = dict(subparsers.choices)
    found["batch"] = batch._parser()
    return found


# --------------------------------------------------------------------------- options and settings

def test_every_option_of_every_command_is_in_the_readme():
    readme = _text(README)
    for name, parser in _parsers().items():
        assert f"edgeserve {name}" in readme, name
        for action in parser._actions:
            for option in action.option_strings:
                if option not in ("-h", "--help"):
                    assert re.search(rf"(?<![\w-]){re.escape(option)}(?![\w-])", readme), f"{name} {option}"


def test_every_setting_is_documented_and_no_unknown_one_is():
    readme = _text(README)
    settings = _section(README, "Options and settings")
    for name in config.SETTINGS:
        assert f"`{name}`" in settings, name
    known = set(config.SETTINGS) | {"EDGESERVE_CONFIG_DIR", "EDGESERVE_SOURCES", "EDGESERVE_BUILD_ROOT"}
    for page in PAGES:
        for name in set(re.findall(r"\bEDGESERVE_[A-Z_]+\b", _text(page))):
            assert name in known, f"{page.name} mentions {name}, which no longer exists"
    assert "EDGESERVE_CONFIG_DIR" in readme


def test_the_documented_defaults_are_the_real_ones():
    reference = _section(README, "Options and settings")
    defaults = {"--max-input-len N": "3072", "--max-batch-size N": "1", "--host ADDRESS": "127.0.0.1",
                "--port N": "8000", "--cache-dir DIR": "`~/.cache/edgeserve`", "--max-new-tokens N":
                "the model's `generation_config.json`, else 512"}
    for option, default in defaults.items():
        assert re.search(rf"\| `{re.escape(option)}` \| `EDGESERVE_[A-Z_]+` \| {re.escape(default)} \|", reference), option
    parser = argparse.ArgumentParser()
    edgeserve.settings.add_engine_arguments(parser)
    edgeserve.settings.add_server_arguments(parser)
    resolved = edgeserve.settings.resolve(parser.parse_args([]))
    assert (resolved["max_input_len"][0], resolved["max_batch_size"][0], resolved["host"][0], resolved["port"][0]) \
        == (3072, 1, "127.0.0.1", 8000)


# --------------------------------------------------------------------------- commands in the docs

def _edgeserve_commands(page: Path, inline: bool = False) -> list[list[str]]:
    """``edgeserve`` command lines from the page's bash blocks, or with ``inline`` from its inline code (where
    they may be mentions such as `edgeserve batch`; syntax summaries such as `edgeserve setup [--network]` are
    left out)."""
    commands = []
    sources = ([span for span in re.findall(r"`(edgeserve [^`]+)`", _text(page)) if not re.search(r"[\[\]…|]", span)]
               if inline else _blocks(page, "bash"))
    for block in sources:
        for line in block.replace("\\\n", " ").splitlines():
            line = line.split(" #", 1)[0].strip()
            if not re.search(r"(^|/|\s)edgeserve(\s|$)", line) or line.startswith("#"):
                continue  # not an edgeserve command (and maybe half of a multi-line quoted string)
            words = shlex.split(re.sub(r"<[^>]+>", "x", line))
            while words and re.fullmatch(r"[A-Z_]+=\S*", words[0]):  # VAR=value edgeserve ...
                words = words[1:]
            if words and Path(words[0]).name == "edgeserve":
                args = words[1:]
                commands.append(args[:args.index("--")] if "--" in args else args)  # after --: the exporter's
    return commands


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_every_edgeserve_command_in_the_docs_parses(page):
    parsers = _parsers()
    for args in _edgeserve_commands(page):
        if args == ["--version"]:
            continue
        name, rest = args[0], args[1:]
        assert name in parsers, f"{page.name}: unknown command edgeserve {name}"
        try:
            parsers[name].parse_args(rest)
        except SystemExit as exc:
            pytest.fail(f"{page.name}: `edgeserve {' '.join(args)}` does not parse ({exc})")


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_every_edgeserve_command_mentioned_in_the_text_exists(page):
    parsers = _parsers()
    for args in _edgeserve_commands(page, inline=True):
        if args == ["--version"]:
            continue
        assert args[0] in parsers, f"{page.name}: unknown command edgeserve {args[0]}"
        known = {option for action in parsers[args[0]]._actions for option in action.option_strings}
        for word in args[1:]:
            if word.startswith("--"):
                assert word in known, f"{page.name}: edgeserve {args[0]} has no option {word}"


def test_the_readme_shows_every_command():
    shown = {args[0] for args in [*_edgeserve_commands(README), *_edgeserve_commands(README, inline=True)]
             if args and args[0] != "--version"}
    assert shown == set(_parsers()), set(_parsers()) - shown


# --------------------------------------------------------------------------- nothing stale

STALE = ["scripts/", "third_party", "build/edgellm", "smoke_test", "edgeserve-run", "install_service",
         "MODEL_NOTES", "setup.sh", "build_edgellm.sh", "EDGESERVE_URL", "EDGESERVE_CA_CERT"]


def test_nothing_refers_to_removed_files_or_settings():
    sources = [*PAGES, *sorted((REPO / "edgeserve").rglob("*.py")), *nativebuild.source_files()]
    for path in sources:
        if path.suffix == ".patch":  # Edge-LLM's own files
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for word in STALE:
            assert word not in text, f"{path.relative_to(REPO)} mentions {word}"


# Addresses a document may show: loopback, "every interface", private networks, documentation examples (RFC 5737).
_EXAMPLE_ADDRESS = re.compile(r"^(127\.|0\.0\.0\.0$|10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|192\.0\.2\.|198\.51\.100\.|"
                              r"203\.0\.113\.)")


def test_the_docs_are_general_not_about_one_deployment():
    for page in [README, *DOCS]:
        text = _text(page)
        for word in ("spk_", "transcript", "reduced_vocab_model", "/home/"):
            assert word not in text, f"{page.name} mentions {word!r}"
        for address in re.findall(r"(?<![\d.])\d{1,3}(?:\.\d{1,3}){3}(?![\d.])", text):
            assert _EXAMPLE_ADDRESS.match(address), f"{page.name} shows the address {address}, not an example one"


# --------------------------------------------------------------------------- references and links

def test_every_section_the_code_refers_to_exists():
    readme, docs = _headings(README), {page.name: _headings(page) for page in DOCS}
    sources = [*sorted((REPO / "edgeserve").rglob("*.py")), nativebuild.sources_dir() / "CMakeLists.txt"]
    found = 0
    for path in sources:
        text = re.sub(r'"\s*\n\s*(f?)"', "", path.read_text())  # join implicitly concatenated strings
        for match in re.finditer(r"README, '([^']+)'|README, (Troubleshooting)", text):
            assert (match.group(1) or match.group(2)) in readme, f"{path.name}: README has no {match.group(0)!r}"
            found += 1
        for match in re.finditer(r"docs/(\w+\.md), '([^']+)'", text):
            assert match.group(2) in docs[match.group(1)], f"{path.name}: {match.group(0)!r} does not exist"
            found += 1
    assert found >= 5


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_every_link_points_to_an_existing_file_and_section(page):
    text = _text(page)
    links = [(m.group(1), m.group(2)) for m in GITHUB.finditer(text)]
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        path, _, anchor = target.partition("#")
        links.append((str((page.parent / path).resolve().relative_to(REPO)) if path else
                      str(page.relative_to(REPO)), f"#{anchor}" if anchor else None))
    for path, anchor in links:
        assert (REPO / path).exists(), f"{page.name} links to {path}, which does not exist"
        if anchor:
            anchors = {_anchor(h) for h in _headings(REPO / path)}
            assert anchor[1:] in anchors, f"{page.name} links to {path}{anchor}, which has no such section"


def test_the_github_links_and_the_package_metadata_name_the_same_repository():
    owners = set(re.findall(r"https://github\.com/([^/]+)/edgeserve", _text(README)))
    pyproject = (REPO / "pyproject.toml").read_text()
    assert owners and owners == set(re.findall(r"https://github\.com/([^/]+)/edgeserve", pyproject))


# --------------------------------------------------------------------------- versions

def test_the_documented_edgellm_commit_and_versions_match():
    commit = nativebuild.edgellm_commit()
    for page in PAGES:
        for match in re.findall(r"(?:commit|checkout) `?([0-9a-f]{7,40})`?", _text(page)):
            assert commit.startswith(match), f"{page.name} names Edge-LLM commit {match}, not {commit[:12]}"
    major_minor = ".".join(edgeserve.__version__.split(".")[:2])
    edgellm = ".".join(nativebuild.EDGELLM_VERSION.split(".")[:2])
    assert f"edgeserve {major_minor} is built on TensorRT Edge-LLM {edgellm}" in _text(README)
    assert f"git checkout {commit[:12]}" in _text(REPO / "docs/EXPORT.md")  # the manual tools install


def test_the_documented_int4_gpus_are_the_ones_with_kernels():
    sentence = re.search(r"They exist for sm_((?:\d+|,|\s|and)+)", _text(REPO / "docs" / "MODELS.md")).group(1)
    assert {int(n) for n in re.findall(r"\d+", sentence)} == set(INT4_PLUGIN_SMS)


# --------------------------------------------------------------------------- the README's examples, run

@pytest.fixture
def example_model(tmp_path) -> Path:
    """A model folder named like the README's example, as the exporter lays it out (<export>/llm)."""
    return make_tiny_model(tmp_path / "qwen2.5-0.5b-instruct-awq", name="llm")


def _run_bash(script: str, cwd: Path, **env: str) -> str:
    result = subprocess.run(["bash", "-euo", "pipefail", "-c", script], cwd=cwd, capture_output=True, text=True,
                            timeout=120, env={**os.environ, **env})
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_the_readme_api_examples_work(example_model, fake_native, tmp_path):
    if shutil.which("curl") is None:
        pytest.skip("needs curl")
    section = _section(README, "5. Send requests")
    curl, = [b for b in _blocks(README, "bash") if b in section]
    python, = [b for b in _blocks(README, "python") if b in section]
    with edgeserve_server(example_model, fake_native) as url:
        out = _run_bash(curl.replace("http://127.0.0.1:8000", url), tmp_path).replace(" ", "")
        assert '"id":"qwen2.5-0.5b-instruct-awq"' in out  # the name the README tells the SDK example to use
        assert '"request_id":"demo-001"' in out
        script = tmp_path / "example.py"
        script.write_text(python.replace("http://127.0.0.1:8000", url))
        result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().endswith("tokens/s")


def test_the_readme_network_examples_work(example_model, fake_native, tmp_path):
    if shutil.which("curl") is None:
        pytest.skip("needs curl")
    section = _section(README, "6. Access from other machines")
    client_block = next(b for b in _blocks(README, "bash") if b in section and "curl --cacert" in b)
    python, = [b for b in _blocks(README, "python") if b in section]
    with edgeserve_server(example_model, fake_native, network=True) as url:
        key = config.read()["EDGESERVE_API_KEY"]
        shutil.copy(config.config_dir() / "edgeserve-ca.crt", tmp_path / "edgeserve-ca.crt")
        # every curl command of the Linux/macOS block (scp and the key come from the test instead)
        curls = "\n".join(line for line in client_block.splitlines() if not line.startswith(("scp ", "export ")))
        out = _run_bash(curls.replace("https://<server-ip>:8000", url), tmp_path, EDGESERVE_API_KEY=key)
        assert "qwen2.5-0.5b-instruct-awq" in out
        assert '"request_id":"client-001"' in out.replace(" ", "")  # the chat request, with its ID
        script = tmp_path / "client.py"
        script.write_text(python.replace("https://<server-ip>:8000", url)
                          + "\nprint(client.models.list().data[0].id)\n")
        result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120, cwd=tmp_path,
                                env={**os.environ, "EDGESERVE_API_KEY": key})
        assert result.returncode == 0, result.stderr
        lines = result.stdout.strip().splitlines()
        assert lines[0].startswith("client-003 ") and lines[-1] == "qwen2.5-0.5b-instruct-awq"
