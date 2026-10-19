"""What the wheel and sdist must contain (the native sources edgeserve setup builds), and one version everywhere."""

from __future__ import annotations

import fnmatch
import importlib.metadata
import re
import sys

import pytest

import edgeserve
from edgeserve import nativebuild

from conftest import REPO

if sys.version_info >= (3, 11):
    import tomllib
else:
    tomllib = pytest.importorskip("tomli")

PYPROJECT = REPO / "pyproject.toml"
pytestmark = pytest.mark.skipif(not PYPROJECT.is_file(), reason="needs the source tree")


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text())


def test_every_native_source_and_exporter_patch_is_package_data():
    from edgeserve import exporter

    patterns = _pyproject()["tool"]["setuptools"]["package-data"]["edgeserve"]
    root = nativebuild.sources_dir().parent
    files = [*nativebuild.source_files(), *exporter.patch_files()]
    assert len(exporter.patch_files()) == 2
    for path in files:
        relative = path.relative_to(root).as_posix()
        assert any(fnmatch.fnmatch(relative, pattern) for pattern in patterns), f"{relative} would not be packaged"


def test_the_version_has_one_source():
    project = _pyproject()["project"]
    assert "version" not in project and project["dynamic"] == ["version"]
    assert _pyproject()["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "edgeserve.__version__"}
    assert importlib.metadata.version("edgeserve") == edgeserve.__version__
    assert re.fullmatch(r"\d+\.\d+\.\d+", edgeserve.__version__)


def test_the_metadata_names_the_license_and_the_command():
    project = _pyproject()["project"]
    assert project["license"] == "Apache-2.0" and set(project["license-files"]) == {"LICENSE", "NOTICE"}
    assert project["scripts"] == {"edgeserve": "edgeserve.cli:main"}
    assert (REPO / "LICENSE").read_text().startswith("\n                                 Apache License")
    assert all(not url.endswith("/") for url in project["urls"].values())


def test_the_sdist_leaves_out_models_outputs_and_secrets():
    manifest = (REPO / "MANIFEST.in").read_text()
    for line in ("prune models", "prune outputs", "prune build"):
        assert line in manifest.splitlines()
    assert "*.key" in manifest and "*.env" in manifest


def test_the_pinned_edgellm_commit_is_a_full_hash():
    assert re.fullmatch(r"[0-9a-f]{40}", nativebuild.edgellm_commit())
