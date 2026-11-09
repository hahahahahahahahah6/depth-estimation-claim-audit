"""CLI smoke tests for depth-estimation-claim-audit."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

from depth_estimation_claim_audit import __version__
from depth_estimation_claim_audit.cli import build_parser, main

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GOOD_CODE = (
    "from transformers import pipeline\n"
    "\n"
    "\n"
    "def run(image):\n"
    '    estimator = pipeline("depth-estimation")\n'
    "    return estimator(image)\n"
)

MIDAS_CODE = (
    "import torch\n"
    "\n"
    'midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")\n'
    "midas.eval()\n"
    "\n"
    "\n"
    "def depth(batch):\n"
    "    return midas(batch)\n"
)

NO_DEPTH_CODE = (
    "from transformers import pipeline\n"
    "\n"
    'detector = pipeline("object-detection")\n'
    "\n"
    "\n"
    "def run(image):\n"
    "    return detector(image)\n"
)

CLAIM_DOC = (
    "# Scene Tool\n"
    "\n"
    "Scene Tool performs monocular depth estimation on every frame.\n"
    "It returns a dense depth map you can feed to the renderer.\n"
)

DENIAL_DOC = (
    "# Scene Tool\n"
    "\n"
    "Scene Tool only draws bounding boxes.\n"
    "It does not perform depth estimation.\n"
)

NEUTRAL_DOC = "# Scene Tool\n\nA small helper for batch image jobs.\n"


def write_project(root, doc: str, modules: dict[str, str]):
    (root / "README.md").write_text(doc, encoding="utf-8")
    for relative, source in modules.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return root


@pytest.fixture()
def good_project(tmp_path):
    return write_project(tmp_path, CLAIM_DOC, {"scene/depth.py": GOOD_CODE})


@pytest.fixture()
def bad_project(tmp_path):
    return write_project(tmp_path, CLAIM_DOC, {"scene/detect.py": NO_DEPTH_CODE})


@pytest.fixture()
def denial_project(tmp_path):
    return write_project(tmp_path, DENIAL_DOC, {"scene/depth.py": MIDAS_CODE})


@pytest.fixture()
def testonly_project(tmp_path):
    return write_project(
        tmp_path,
        CLAIM_DOC,
        {"scene/detect.py": NO_DEPTH_CODE, "tests/test_depth.py": GOOD_CODE},
    )


def run_cli(*argv) -> int:
    return main([str(arg) for arg in argv])


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert args.paths == ["."]
    assert args.fail_on == "high"
    assert args.json is False
    assert args.quiet is False


def test_parser_accepts_paths():
    args = build_parser().parse_args(["a", "b"])
    assert args.paths == ["a", "b"]


def test_parser_repeatable_options():
    args = build_parser().parse_args(
        ["--docs", "a.md", "--docs", "b.md", "--code", "c.py", "--exclude", "*.tmp"]
    )
    assert args.docs == ["a.md", "b.md"]
    assert args.code == ["c.py"]
    assert args.exclude == ["*.tmp"]


def test_parser_rejects_unknown_fail_on():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--fail-on", "critical"])


def test_version_action_prints_version(capsys):
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == __version__


def test_help_mentions_exit_codes(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--help"])
    assert "Exit status" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Smoke: good project
# --------------------------------------------------------------------------- #


def test_good_project_passes(good_project, capsys):
    assert run_cli(good_project) == 0
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "FAIL" not in out


def test_good_project_passes_at_every_threshold(good_project):
    for level in ("none", "low", "medium", "high"):
        assert run_cli(good_project, "--fail-on", level) == 0


def test_good_project_reports_evidence(good_project, capsys):
    run_cli(good_project)
    out = capsys.readouterr().out
    assert "evidence (code)" in out
    assert "hf-pipeline-depth-estimation" in out


def test_good_project_json_status(good_project, capsys):
    assert run_cli(good_project, "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "pass"
    assert data["findings"] == []
    assert data["summary"]["evidence"] == 1


# --------------------------------------------------------------------------- #
# Smoke: bad project
# --------------------------------------------------------------------------- #


def test_bad_project_fails(bad_project, capsys):
    assert run_cli(bad_project) == 2
    assert "FAIL" in capsys.readouterr().out


def test_bad_project_reports_exactly_one_finding(bad_project, capsys):
    run_cli(bad_project, "--json")
    data = json.loads(capsys.readouterr().out)
    assert [finding["id"] for finding in data["findings"]] == [
        "documented-depth-estimation-claim-false"
    ]
    assert data["findings"][0]["severity"] == "high"


def test_bad_project_text_report_names_the_finding(bad_project, capsys):
    run_cli(bad_project)
    out = capsys.readouterr().out
    assert "[HIGH] documented-depth-estimation-claim-false" in out
    assert "README.md:3" in out


def test_bad_project_passes_when_fail_on_none(bad_project):
    assert run_cli(bad_project, "--fail-on", "none") == 0


def test_bad_project_quiet_hides_summary(bad_project, capsys):
    assert run_cli(bad_project, "--quiet") == 2
    out = capsys.readouterr().out
    assert out.startswith("FAIL:")
    assert "doc file(s)" not in out
    assert "near-misses:" not in out
    assert "documented-depth-estimation-claim-false" in out


# --------------------------------------------------------------------------- #
# Smoke: denial project
# --------------------------------------------------------------------------- #


def test_denial_project_fails(denial_project, capsys):
    assert run_cli(denial_project) == 2
    assert "denied-depth-estimation-contradicted" in capsys.readouterr().out


def test_denial_project_json(denial_project, capsys):
    run_cli(denial_project, "--json")
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "fail"
    assert data["max_severity"] == "high"
    assert [finding["id"] for finding in data["findings"]] == [
        "denied-depth-estimation-contradicted"
    ]
    assert data["denials"][0]["phrase"] == "does-not-perform-depth-estimation"


# --------------------------------------------------------------------------- #
# Smoke: test-only project
# --------------------------------------------------------------------------- #


def test_testonly_project_is_medium(testonly_project, capsys):
    assert run_cli(testonly_project) == 0
    out = capsys.readouterr().out
    assert "[MEDIUM] documented-depth-estimation-claim-test-only" in out


def test_testonly_project_fails_on_medium(testonly_project):
    assert run_cli(testonly_project, "--fail-on", "medium") == 2


def test_testonly_project_fails_on_low(testonly_project):
    assert run_cli(testonly_project, "--fail-on", "low") == 2


def test_testonly_project_json_severity(testonly_project, capsys):
    run_cli(testonly_project, "--json")
    data = json.loads(capsys.readouterr().out)
    assert data["max_severity"] == "medium"
    assert data["evidence"][0]["is_test"] is True


# --------------------------------------------------------------------------- #
# Options
# --------------------------------------------------------------------------- #


def test_exclude_skips_evidence(good_project, capsys):
    assert run_cli(good_project, "--exclude", "*/scene/*") == 2
    assert "documented-depth-estimation-claim-false" in capsys.readouterr().out


def test_explicit_docs_and_code(good_project, capsys):
    code = good_project / "scene" / "depth.py"
    doc = good_project / "README.md"
    assert run_cli("--docs", doc, "--code", code) == 0
    data = capsys.readouterr().out
    assert "scanned 1 doc file(s), 1 python file(s)" in data


def test_explicit_code_only_reports_undocumented(good_project, capsys):
    code = good_project / "scene" / "depth.py"
    assert run_cli("--code", code) == 0
    out = capsys.readouterr().out
    assert "[LOW] undocumented-depth-estimation" in out


def test_explicit_code_only_fails_on_low(good_project):
    code = good_project / "scene" / "depth.py"
    assert run_cli("--code", code, "--fail-on", "low") == 2


def test_json_output_is_valid_json(good_project, capsys):
    run_cli(good_project, "--json")
    data = json.loads(capsys.readouterr().out)
    assert set(
        ["tool", "version", "fail_on", "status", "exit_code", "summary", "findings"]
    ) <= set(data)


def test_json_reports_requested_threshold(bad_project, capsys):
    run_cli(bad_project, "--json", "--fail-on", "none")
    data = json.loads(capsys.readouterr().out)
    assert data["fail_on"] == "none"
    assert data["exit_code"] == 0
    assert data["status"] == "pass"
    assert len(data["findings"]) == 1


def test_missing_path_is_usage_error(tmp_path, capsys):
    assert run_cli(tmp_path / "nope") == 1
    assert "error:" in capsys.readouterr().err


def test_empty_directory_passes(tmp_path, capsys):
    assert run_cli(tmp_path) == 0
    assert "PASS" in capsys.readouterr().out


def test_near_miss_is_reported_in_text(tmp_path, capsys):
    write_project(
        tmp_path, NEUTRAL_DOC, {"scene/depth.py": 'p = pipeline("depth-estimation")\n'}
    )
    assert run_cli(tmp_path) == 0
    out = capsys.readouterr().out
    assert "near-miss" in out
    assert "PASS" in out


# --------------------------------------------------------------------------- #
# Process-level smoke (module entry point and console script)
# --------------------------------------------------------------------------- #


def run_module(*argv, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "depth_estimation_claim_audit", *[str(a) for a in argv]],
        capture_output=True,
        text=True,
        cwd=cwd or REPO_ROOT,
        env={**os.environ, "PYTHONPATH": REPO_ROOT},
    )


def test_module_entry_point_version():
    completed = run_module("--version")
    assert completed.returncode == 0
    assert completed.stdout.strip() == __version__


def test_module_entry_point_good_project(good_project):
    completed = run_module(good_project)
    assert completed.returncode == 0
    assert "PASS" in completed.stdout


def test_module_entry_point_bad_project(bad_project):
    completed = run_module(bad_project)
    assert completed.returncode == 2
    assert "documented-depth-estimation-claim-false" in completed.stdout


def test_module_entry_point_json(good_project):
    completed = run_module(good_project, "--json")
    assert json.loads(completed.stdout)["status"] == "pass"


CONSOLE_SCRIPT = shutil.which("depth-estimation-claim-audit")


@pytest.mark.skipif(CONSOLE_SCRIPT is None, reason="package is not installed")
def test_console_script_version():
    completed = subprocess.run(
        [CONSOLE_SCRIPT, "--version"], capture_output=True, text=True
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == __version__


@pytest.mark.skipif(CONSOLE_SCRIPT is None, reason="package is not installed")
def test_console_script_bad_project(bad_project):
    completed = subprocess.run(
        [CONSOLE_SCRIPT, str(bad_project)], capture_output=True, text=True
    )
    assert completed.returncode == 2
    assert "documented-depth-estimation-claim-false" in completed.stdout


# --------------------------------------------------------------------------- #
# Self-audit: this project documents no depth estimation of its own
# --------------------------------------------------------------------------- #


def test_self_audit_of_own_readme_is_honest():
    """The README talks *about* depth estimation but denies performing it, so
    the audit must not accuse it of a false capability claim."""
    readme = os.path.join(REPO_ROOT, "README.md")
    package = os.path.join(REPO_ROOT, "depth_estimation_claim_audit")
    assert run_cli("--docs", readme, "--code", package, "--fail-on", "high") == 0


def test_self_audit_finds_no_evidence_in_own_sources(capsys):
    package = os.path.join(REPO_ROOT, "depth_estimation_claim_audit")
    run_cli("--code", package, "--json")
    data = json.loads(capsys.readouterr().out)
    assert data["summary"]["evidence"] == 0
