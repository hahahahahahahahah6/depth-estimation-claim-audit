"""Command line interface for depth-estimation-claim-audit."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from . import __version__
from .audit import (
    FAIL_ON_CHOICES,
    AuditError,
    AuditResult,
    audit_paths,
)

__all__ = ["build_parser", "format_report", "main"]

PROG = "depth-estimation-claim-audit"

_SEVERITY_LABEL = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Audit documentation claims about image depth estimation against the "
            "depth-estimation operations that Python sources actually perform. "
            "Offline, static, read-only."
        ),
        epilog=(
            "Exit status: 0 nothing at or above the --fail-on threshold, "
            "2 findings at or above it, 1 bad usage or unreadable input."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        metavar="PATH",
        help="files or directories to audit; classified by suffix (default: .)",
    )
    parser.add_argument(
        "--docs",
        action="append",
        default=[],
        metavar="PATH",
        help="treat PATH as documentation (repeatable)",
    )
    parser.add_argument(
        "--code",
        action="append",
        default=[],
        metavar="PATH",
        help="treat PATH as Python source (repeatable)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help="skip paths matching GLOB (repeatable)",
    )
    parser.add_argument(
        "--fail-on",
        choices=FAIL_ON_CHOICES,
        default="high",
        help="lowest severity that makes the run fail (default: high)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the full result as JSON instead of text",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print findings only, no scan summary",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="print the version and exit",
    )
    return parser


def _format_hit_line(label: str, path: str, line: int, detail: str, text: str) -> str:
    text = text.strip()
    if len(text) > 120:
        text = text[:117] + "..."
    return f"      {label} {path}:{line} [{detail}] {text}"


def format_report(result: AuditResult, fail_on: str = "high", quiet: bool = False) -> str:
    """Render a human-readable report."""
    lines: list[str] = []
    summary = result.summary()
    if not quiet:
        lines.append(f"{PROG} {__version__}")
        lines.append(
            f"scanned {summary['doc_files']} doc file(s), "
            f"{summary['code_files']} python file(s)"
        )
        lines.append(
            f"claims: {summary['claims']}  denials: {summary['denials']}  "
            f"evidence: {summary['evidence']}  near-misses: {summary['near_misses']}"
        )
        for hit in result.evidence:
            scope = "test" if hit.is_test else "code"
            lines.append(
                f"  evidence ({scope}) {hit.path}:{hit.line} "
                f"[{hit.pattern}] {hit.symbol} -> {hit.invocation}"
            )
        for miss in result.near_misses:
            lines.append(
                f"  near-miss {miss.path}:{miss.line} "
                f"[{miss.pattern}] {miss.symbol} ({miss.reason})"
            )
        lines.append("")

    if not result.findings:
        lines.append("PASS: no depth-estimation claim mismatch found.")
        return "\n".join(lines)

    count = len(result.findings)
    if result.status(fail_on) == "fail":
        lines.append(f"FAIL: {count} finding(s) at or above --fail-on {fail_on}")
    else:
        lines.append(f"PASS: {count} finding(s), all below --fail-on {fail_on}")
    for finding in result.findings:
        lines.append("")
        lines.append(f"  [{_SEVERITY_LABEL[finding.severity]}] {finding.id}")
        lines.append(f"    {finding.title}")
        lines.append(f"    {finding.detail}")
        for claim in finding.claims:
            lines.append(_format_hit_line("claim:", claim.path, claim.line, claim.phrase, claim.text))
        for denial in finding.denials:
            lines.append(
                _format_hit_line("denial:", denial.path, denial.line, denial.phrase, denial.text)
            )
        for hit in finding.evidence:
            lines.append(
                _format_hit_line("code:", hit.path, hit.line, hit.pattern, hit.snippet)
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Returns the process exit code."""
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw)
    # --docs/--code on their own mean "audit exactly these", not "these plus .".
    paths = list(args.paths) if (args.paths or not (args.docs or args.code)) else []
    if (args.docs or args.code) and not _has_positional(raw):
        paths = []
    try:
        result = audit_paths(
            paths,
            doc_paths=args.docs,
            code_paths=args.code,
            exclude=args.exclude,
        )
    except AuditError as exc:
        print(f"{PROG}: error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(args.fail_on), indent=2, sort_keys=False))
    else:
        print(format_report(result, args.fail_on, quiet=args.quiet))
    return result.exit_code(args.fail_on)


def _has_positional(argv: Sequence[str]) -> bool:
    """True when ``argv`` carries at least one positional PATH argument."""
    expect_value = False
    for token in argv:
        if expect_value:
            expect_value = False
            continue
        if token in ("--docs", "--code", "--exclude", "--fail-on"):
            expect_value = True
            continue
        if token.startswith("-"):
            continue
        return True
    return False
