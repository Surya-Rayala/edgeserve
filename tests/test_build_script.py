"""Parts of the packaged edgeserve/native/build.sh that can run without CUDA: its syntax, the TensorRT version it
reads from the headers, and the source step (pinned checkout + patches) on scratch git repositories."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from edgeserve.nativebuild import sources_dir

BUILD_SH = sources_dir() / "build.sh"


def _clean_env(**extra: str) -> dict[str, str]:
    env = {name: value for name, value in os.environ.items() if not name.startswith("EDGESERVE_")}
    env.update(extra)
    return env


def test_the_build_script_is_valid_bash():
    result = subprocess.run(["bash", "-n", str(BUILD_SH)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


_TRT_10_3 = """#define NV_TENSORRT_MAJOR 10 //!< TensorRT major version.
#define NV_TENSORRT_MINOR 3 //!< TensorRT minor version.
#define NV_TENSORRT_PATCH 0 //!< TensorRT patch version.
#define NV_TENSORRT_BUILD 30 //!< TensorRT build number.
#define NV_TENSORRT_LWS_MAJOR 0 //!< TensorRT LWS major version.
"""
_TRT_10_16 = """#define TRT_MAJOR_ENTERPRISE 10
#define TRT_MINOR_ENTERPRISE 16
#define TRT_PATCH_ENTERPRISE 0
#define TRT_BUILD_ENTERPRISE 12
#define NV_TENSORRT_MAJOR TRT_MAJOR_ENTERPRISE //!< TensorRT major version.
#define NV_TENSORRT_MINOR TRT_MINOR_ENTERPRISE //!< TensorRT minor version.
#define NV_TENSORRT_PATCH TRT_PATCH_ENTERPRISE //!< TensorRT patch version.
#define NV_TENSORRT_BUILD TRT_BUILD_ENTERPRISE //!< TensorRT build number.
"""


@pytest.mark.parametrize(("header", "version"), [(_TRT_10_3, "10.3.0.30"), (_TRT_10_16, "10.16.0.12")])
def test_the_build_script_reads_the_tensorrt_version_of_old_and_new_headers(tmp_path, header, version):
    awk = shutil.which("awk")
    if awk is None:
        pytest.skip("awk is not installed")
    match = re.search(r"TRT_VER=\"\$\(awk '(.*?)' \"\$TRT_HDR\"\)\"", BUILD_SH.read_text(), re.S)
    assert match, "build.sh no longer reads the TensorRT version with one awk program"
    (tmp_path / "NvInferVersion.h").write_text(header)
    result = subprocess.run([awk, match.group(1), str(tmp_path / "NvInferVersion.h")],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0 and result.stdout == version


def test_the_build_script_refuses_paths_with_spaces(tmp_path):
    result = subprocess.run(["bash", str(BUILD_SH)], capture_output=True, text=True, timeout=60,
                            env=_clean_env(EDGESERVE_SOURCES=str(sources_dir()),
                                           EDGESERVE_BUILD_ROOT=str(tmp_path / "build root")))
    assert result.returncode == 1 and "paths with spaces" in result.stderr


@pytest.fixture
def edgellm_checkout(tmp_path):
    """An "upstream" repository with two commits, a clone of it, and two patches; run(commit) runs the source step
    of build.sh (pinned checkout + patches) on the clone."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    env = _clean_env(HOME=str(tmp_path), GIT_CONFIG_NOSYSTEM="1", GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
                     GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")

    def git(*args: str, cwd: Path) -> str:
        return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True,
                              timeout=60).stdout.strip()

    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git("init", "-q", cwd=upstream)
    (upstream / "CMakeLists.txt").write_text("project(edgellm)\nadd_subdirectory(cpp)\n")
    (upstream / "plugin.cpp").write_text("int opt = 0;\nint max = 1;\n")
    git("add", ".", cwd=upstream)
    git("commit", "-qm", "v1", cwd=upstream)
    v1 = git("rev-parse", "HEAD", cwd=upstream)
    with open(upstream / "CMakeLists.txt", "a") as f:
        f.write("install()\n")
    git("commit", "-qam", "v2", cwd=upstream)
    v2 = git("rev-parse", "HEAD", cwd=upstream)
    src = tmp_path / "src"
    git("clone", "-q", str(upstream), str(src), cwd=tmp_path)
    patches = tmp_path / "patches"
    patches.mkdir()

    def make_patch(name: str, file: str, old: str, new: str) -> None:
        """A patch of ``file`` as it is in the checkout now (which must be v1's content)."""
        path = src / file
        original = path.read_text()
        path.write_text(original.replace(old, new))
        (patches / name).write_text(git("diff", "--", file, cwd=src) + "\n")
        path.write_text(original)

    git("checkout", "-q", v1, cwd=src)  # the patches are made for v1; they also apply to v2
    make_patch("0001-opt.patch", "plugin.cpp", "int opt = 0;", "int opt = max;")
    make_patch("0002-languages.patch", "CMakeLists.txt", "project(edgellm)", "project(edgellm LANGUAGES CXX)")
    git("checkout", "-q", "-", cwd=src)  # back at v2, as a fresh clone is
    text = BUILD_SH.read_text()
    step = text[text.index('if [[ "$(git -C "$SRC" rev-parse HEAD)" != "$COMMIT" ]]; then'):
                text.index("# ---------------------------------------------------------------- CuTe DSL kernel pack")]

    def run(commit: str) -> subprocess.CompletedProcess:
        script = ("set -euo pipefail\n"
                  f"SRC={shlex.quote(str(src))} COMMIT={commit} PATCH_DIR={shlex.quote(str(patches))}\n"
                  "log() { printf '[edgeserve-build] %s\\n' \"$*\"; }\n"
                  "die() { printf '[edgeserve-build] ERROR: %s\\n' \"$*\" >&2; exit 1; }\n" + step)
        return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=120)

    return SimpleNamespace(src=src, patches=patches, v1=v1, v2=v2, run=run, make_patch=make_patch,
                           head=lambda: git("rev-parse", "HEAD", cwd=src))


def test_the_patches_are_applied_to_the_pinned_commit_once(edgellm_checkout):
    box = edgellm_checkout
    first = box.run(box.v1)
    assert first.returncode == 0, first.stderr
    assert box.head() == box.v1
    assert "applied 0001-opt.patch" in first.stdout and "applied 0002-languages.patch" in first.stdout
    assert (box.src / "plugin.cpp").read_text().startswith("int opt = max;")
    again = box.run(box.v1)
    assert again.returncode == 0 and "applied" not in again.stdout  # nothing touched: no rebuild


def test_an_updated_patch_is_applied_again(edgellm_checkout):
    box = edgellm_checkout
    assert box.run(box.v1).returncode == 0
    (box.src / "plugin.cpp").write_text("int opt = 0;\nint max = 1;\n")  # as at v1, to diff against
    box.make_patch("0001-opt.patch", "plugin.cpp", "int opt = 0;", "int opt = max + 0;")
    (box.src / "plugin.cpp").write_text("int opt = max;\nint max = 1;\n")  # the old version is what is applied
    updated = box.run(box.v1)
    assert updated.returncode == 0, updated.stderr
    assert (box.src / "plugin.cpp").read_text() == "int opt = max + 0;\nint max = 1;\n"
    assert (box.src / "CMakeLists.txt").read_text().startswith("project(edgellm LANGUAGES CXX)")


def test_a_patched_checkout_moves_to_a_new_pinned_commit(edgellm_checkout):
    box = edgellm_checkout
    assert box.run(box.v1).returncode == 0  # the patched files differ from v1, and v2 changes one of them
    bumped = box.run(box.v2)
    assert bumped.returncode == 0, bumped.stderr
    assert box.head() == box.v2
    assert (box.src / "CMakeLists.txt").read_text() == "project(edgellm LANGUAGES CXX)\nadd_subdirectory(cpp)\ninstall()\n"
    assert (box.src / "plugin.cpp").read_text().startswith("int opt = max;")


def test_a_patch_that_does_not_apply_stops_the_build(edgellm_checkout):
    box = edgellm_checkout
    (box.patches / "0003-broken.patch").write_text("--- a/nothing.txt\n+++ b/nothing.txt\n@@ -1 +1 @@\n-a\n+b\n")
    result = box.run(box.v1)
    assert result.returncode == 1 and "patch does not apply: 0003-broken.patch" in result.stderr
