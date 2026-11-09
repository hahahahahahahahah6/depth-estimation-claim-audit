"""Core audit logic: read docs, read code, compare, report.

The audit answers one question: *does the documentation's story about image
depth estimation match what the Python sources actually do?*

Three kinds of signal are collected, all by static text matching:

``claims``
    Prose sentences in documentation that advertise depth estimation.
``denials``
    Prose sentences that explicitly state depth estimation is **not** done.
``evidence``
    Depth estimators that are both constructed **and** invoked in Python
    source, plus calls to unambiguously depth-named functions.

A constructed-but-never-invoked estimator is recorded as a *near miss*: useful
context for a human, never treated as evidence.
"""

from __future__ import annotations

import bisect
import fnmatch
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .evidence import (
    CLAIM_PATTERNS,
    CODE_SUFFIXES,
    DENIAL_PATTERNS,
    DOC_SUFFIXES,
    INVOCATION_SPECS,
    SKIP_DIRS,
    has_chained_call,
    iter_generic_calls,
    iter_invocations,
    iter_loader_matches,
    line_of,
    mask_string_literals,
    strip_comments,
)

__all__ = [
    "AuditError",
    "AuditResult",
    "ClaimHit",
    "DenialHit",
    "EvidenceHit",
    "FAIL_ON_CHOICES",
    "FINDING_IDS",
    "Finding",
    "NearMiss",
    "SEVERITIES",
    "SEVERITY_RANK",
    "audit_paths",
    "audit_texts",
    "build_findings",
    "collect_code_files",
    "collect_doc_files",
    "exit_code_for",
    "is_test_path",
    "scan_code_text",
    "scan_doc_text",
]

SEVERITIES: tuple[str, ...] = ("low", "medium", "high")
SEVERITY_RANK: dict[str, int] = {"low": 1, "medium": 2, "high": 3}
FAIL_ON_CHOICES: tuple[str, ...] = ("none", "low", "medium", "high")

FINDING_IDS: tuple[str, ...] = (
    "documented-depth-estimation-claim-false",
    "denied-depth-estimation-contradicted",
    "documented-depth-estimation-claim-test-only",
    "contradictory-depth-estimation-documentation",
    "undocumented-depth-estimation",
)

_SNIPPET_LIMIT = 200


class AuditError(Exception):
    """Raised for unusable input (missing path, undecodable file)."""


# --------------------------------------------------------------------------- #
# Hit records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ClaimHit:
    """A documentation sentence that advertises depth estimation."""

    path: str
    line: int
    phrase: str
    text: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "line": self.line,
            "phrase": self.phrase,
            "text": self.text,
        }


@dataclass(frozen=True)
class DenialHit:
    """A documentation sentence that denies depth estimation."""

    path: str
    line: int
    phrase: str
    text: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "line": self.line,
            "phrase": self.phrase,
            "text": self.text,
        }


@dataclass(frozen=True)
class EvidenceHit:
    """A depth-estimation operation that is really performed in source."""

    path: str
    line: int
    pattern: str
    backend: str
    symbol: str
    kind: str
    invocation: str
    snippet: str
    is_test: bool
    loader_line: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "line": self.line,
            "pattern": self.pattern,
            "backend": self.backend,
            "symbol": self.symbol,
            "kind": self.kind,
            "invocation": self.invocation,
            "snippet": self.snippet,
            "is_test": self.is_test,
            "loader_line": self.loader_line,
        }


@dataclass(frozen=True)
class NearMiss:
    """A depth estimator that is constructed but never invoked."""

    path: str
    line: int
    pattern: str
    symbol: str
    reason: str
    snippet: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "line": self.line,
            "pattern": self.pattern,
            "symbol": self.symbol,
            "reason": self.reason,
            "snippet": self.snippet,
        }


@dataclass(frozen=True)
class Finding:
    """One audit conclusion."""

    id: str
    severity: str
    title: str
    detail: str
    claims: tuple[ClaimHit, ...] = ()
    denials: tuple[DenialHit, ...] = ()
    evidence: tuple[EvidenceHit, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "claims": [hit.to_dict() for hit in self.claims],
            "denials": [hit.to_dict() for hit in self.denials],
            "evidence": [hit.to_dict() for hit in self.evidence],
        }


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #

_TEST_DIR_NAMES = frozenset({"test", "tests", "testing", "_tests"})


def normalize_path(path: os.PathLike[str] | str) -> str:
    """Posix-style string form of ``path``, for stable output across OSes."""
    return str(os.fspath(path)).replace("\\", "/")


def is_test_path(path: os.PathLike[str] | str) -> bool:
    """True when ``path`` looks like test material rather than shipped code."""
    text = normalize_path(path)
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if not parts:
        return False
    name = parts[-1]
    stem = name.rsplit(".", 1)[0]
    if stem == "conftest" or stem.startswith("test_") or stem.endswith("_test"):
        return True
    return any(part.lower() in _TEST_DIR_NAMES for part in parts[:-1])


def _is_excluded(path: str, exclude: Sequence[str]) -> bool:
    for pattern in exclude:
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern):
            return True
    return False


def _in_skipped_dir(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts[:-1])


def _collect(
    paths: Iterable[os.PathLike[str] | str],
    suffixes: frozenset[str],
    exclude: Sequence[str] = (),
    *,
    force: bool = False,
) -> list[str]:
    collected: list[str] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise AuditError(f"path not found: {normalize_path(raw)}")
        if path.is_file():
            candidates = [path]
        else:
            candidates = sorted(
                child for child in path.rglob("*") if child.is_file()
            )
        for candidate in candidates:
            text = normalize_path(candidate)
            if _in_skipped_dir(candidate):
                continue
            if _is_excluded(text, exclude):
                continue
            explicit_file = candidate == path
            if not (force and explicit_file) and candidate.suffix.lower() not in suffixes:
                continue
            collected.append(text)
    seen: set[str] = set()
    unique: list[str] = []
    for item in collected:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def collect_doc_files(
    paths: Iterable[os.PathLike[str] | str],
    exclude: Sequence[str] = (),
    *,
    force: bool = False,
) -> list[str]:
    """Documentation files under ``paths``."""
    return _collect(paths, DOC_SUFFIXES, exclude, force=force)


def collect_code_files(
    paths: Iterable[os.PathLike[str] | str],
    exclude: Sequence[str] = (),
    *,
    force: bool = False,
) -> list[str]:
    """Python source files under ``paths``."""
    return _collect(paths, CODE_SUFFIXES, exclude, force=force)


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:  # pragma: no cover - depends on filesystem state
        raise AuditError(f"cannot read {path}: {exc}") from exc


def _snippet(lines: Sequence[str], lineno: int) -> str:
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].strip()[:_SNIPPET_LIMIT]
    return ""


# --------------------------------------------------------------------------- #
# Code scanning
# --------------------------------------------------------------------------- #


def scan_code_text(
    text: str, path: str = "<string>", *, is_test: bool | None = None
) -> tuple[list[EvidenceHit], list[NearMiss]]:
    """Find depth-estimation evidence and near misses in one Python source."""
    if is_test is None:
        is_test = is_test_path(path)
    code = strip_comments(text)
    masked = mask_string_literals(code)
    lines = text.splitlines()
    evidence: dict[tuple[object, ...], EvidenceHit] = {}
    near_misses: list[NearMiss] = []

    for pattern, match in iter_loader_matches(code):
        # If the construct sits inside a string literal, the opening paren of
        # its call is blanked in the masked source. Docstrings and sample code
        # held in strings never count as evidence.
        paren = code.find("(", match.start(), match.end())
        if paren == -1 or masked[paren] != "(":
            continue
        groups = match.groupdict()
        spec = INVOCATION_SPECS[pattern.invocation]
        var = groups.get("var")
        symbol = groups.get("symbol") or pattern.symbol or pattern.name
        loader_line = line_of(code, match.start())

        calls = list(iter_invocations(masked, var, spec)) if var else []
        if not calls:
            line_end = masked.find("\n", match.start())
            if line_end == -1:
                line_end = len(masked)
            chained = has_chained_call(masked[match.start() : line_end], spec)
            if chained is not None:
                calls = [(match.start(), chained)]
        if not calls:
            near_misses.append(
                NearMiss(
                    path=path,
                    line=loader_line,
                    pattern=pattern.name,
                    symbol=symbol,
                    reason="loader-without-invocation",
                    snippet=_snippet(lines, loader_line),
                )
            )
            continue

        position, attr = calls[0]
        call_line = line_of(masked, position)
        hit = EvidenceHit(
            path=path,
            line=call_line,
            pattern=pattern.name,
            backend=pattern.backend,
            symbol=symbol,
            kind="loader-and-call",
            invocation=attr,
            snippet=_snippet(lines, call_line),
            is_test=is_test,
            loader_line=loader_line,
        )
        evidence.setdefault((hit.path, hit.line, hit.pattern, hit.symbol, hit.invocation), hit)

    for position, name, owner in iter_generic_calls(masked):
        call_line = line_of(masked, position)
        symbol = f"{owner}.{name}" if owner else name
        hit = EvidenceHit(
            path=path,
            line=call_line,
            pattern="generic-depth-call",
            backend="generic",
            symbol=symbol,
            kind="generic-call",
            invocation=name,
            snippet=_snippet(lines, call_line),
            is_test=is_test,
            loader_line=None,
        )
        evidence.setdefault((hit.path, hit.line, hit.pattern, hit.symbol, hit.invocation), hit)

    ordered = sorted(evidence.values(), key=lambda hit: (hit.line, hit.pattern, hit.symbol))
    return ordered, near_misses


# --------------------------------------------------------------------------- #
# Documentation scanning
# --------------------------------------------------------------------------- #

_FENCE_RE = re.compile(r"^(```+|~~~+)")
_UNIT_START_RE = re.compile(r"^(?:[-*+>|]\s|#{1,6}\s|\d+[.)]\s|\[)")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?;:])\s+")


def _split_block(block: Sequence[tuple[int, str]]) -> list[tuple[str, int]]:
    offsets: list[int] = []
    linenos: list[int] = []
    pieces: list[str] = []
    position = 0
    for lineno, line in block:
        offsets.append(position)
        linenos.append(lineno)
        pieces.append(line)
        position += len(line) + 1
    joined = " ".join(pieces)

    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(joined):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(joined)))

    segments: list[tuple[str, int]] = []
    for begin, end in spans:
        text = joined[begin:end].strip()
        if not text:
            continue
        index = max(0, bisect.bisect_right(offsets, begin) - 1)
        segments.append((text, linenos[index]))
    return segments


def doc_segments(text: str) -> list[tuple[str, int]]:
    """Split documentation into ``(sentence, line_number)`` prose units.

    Fenced code blocks are skipped: the audit judges prose claims, not sample
    code. Headings, list items and table rows are kept as separate units so a
    denial in one bullet never swallows a claim in the next.
    """
    block: list[tuple[int, str]] = []
    segments: list[tuple[str, int]] = []
    fence: str | None = None

    def flush() -> None:
        nonlocal block
        if block:
            segments.extend(_split_block(block))
            block = []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        fence_match = _FENCE_RE.match(stripped)
        if fence is not None:
            if fence_match and stripped.startswith(fence):
                fence = None
            continue
        if fence_match:
            flush()
            fence = fence_match.group(1)[:3]
            continue
        if not stripped:
            flush()
            continue
        if _UNIT_START_RE.match(stripped):
            flush()
        block.append((lineno, stripped))
    flush()
    return segments


def scan_doc_text(text: str, path: str = "<string>") -> tuple[list[ClaimHit], list[DenialHit]]:
    """Find depth-estimation claims and denials in one documentation file.

    A denial wins inside its own sentence: "does not perform depth estimation"
    is a denial, not a claim.
    """
    claims: list[ClaimHit] = []
    denials: list[DenialHit] = []
    for segment, lineno in doc_segments(text):
        snippet = segment[:_SNIPPET_LIMIT]
        denied = False
        for name, regex in DENIAL_PATTERNS:
            if regex.search(segment):
                denials.append(DenialHit(path=path, line=lineno, phrase=name, text=snippet))
                denied = True
                break
        if denied:
            continue
        for name, regex in CLAIM_PATTERNS:
            if regex.search(segment):
                claims.append(ClaimHit(path=path, line=lineno, phrase=name, text=snippet))
                break
    return claims, denials


# --------------------------------------------------------------------------- #
# Result assembly
# --------------------------------------------------------------------------- #


@dataclass
class AuditResult:
    """Everything the audit learned, plus its conclusions."""

    doc_files: list[str] = field(default_factory=list)
    code_files: list[str] = field(default_factory=list)
    claims: list[ClaimHit] = field(default_factory=list)
    denials: list[DenialHit] = field(default_factory=list)
    evidence: list[EvidenceHit] = field(default_factory=list)
    near_misses: list[NearMiss] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    @property
    def real_evidence(self) -> list[EvidenceHit]:
        """Evidence found outside test files."""
        return [hit for hit in self.evidence if not hit.is_test]

    @property
    def test_evidence(self) -> list[EvidenceHit]:
        """Evidence found only in test files."""
        return [hit for hit in self.evidence if hit.is_test]

    def max_severity(self) -> str | None:
        """Highest severity among findings, or ``None`` when clean."""
        if not self.findings:
            return None
        return max(self.findings, key=lambda finding: SEVERITY_RANK[finding.severity]).severity

    def exit_code(self, fail_on: str = "high") -> int:
        """``0`` when nothing at/above ``fail_on`` was found, else ``2``."""
        return exit_code_for(self.findings, fail_on)

    def status(self, fail_on: str = "high") -> str:
        """``"pass"`` or ``"fail"`` for the given threshold."""
        return "fail" if self.exit_code(fail_on) else "pass"

    def summary(self) -> dict[str, int]:
        return {
            "doc_files": len(self.doc_files),
            "code_files": len(self.code_files),
            "claims": len(self.claims),
            "denials": len(self.denials),
            "evidence": len(self.evidence),
            "near_misses": len(self.near_misses),
            "findings": len(self.findings),
        }

    def to_dict(self, fail_on: str = "high") -> dict[str, object]:
        from . import __version__

        return {
            "tool": "depth-estimation-claim-audit",
            "version": __version__,
            "fail_on": fail_on,
            "status": self.status(fail_on),
            "exit_code": self.exit_code(fail_on),
            "max_severity": self.max_severity(),
            "summary": self.summary(),
            "doc_files": list(self.doc_files),
            "code_files": list(self.code_files),
            "claims": [hit.to_dict() for hit in self.claims],
            "denials": [hit.to_dict() for hit in self.denials],
            "evidence": [hit.to_dict() for hit in self.evidence],
            "near_misses": [hit.to_dict() for hit in self.near_misses],
            "findings": [finding.to_dict() for finding in self.findings],
        }


def exit_code_for(findings: Sequence[Finding], fail_on: str = "high") -> int:
    """Exit code for ``findings`` under the ``fail_on`` threshold."""
    if fail_on not in FAIL_ON_CHOICES:
        raise AuditError(f"unknown --fail-on value: {fail_on}")
    if fail_on == "none":
        return 0
    threshold = SEVERITY_RANK[fail_on]
    for finding in findings:
        if SEVERITY_RANK[finding.severity] >= threshold:
            return 2
    return 0


def build_findings(
    claims: Sequence[ClaimHit],
    denials: Sequence[DenialHit],
    evidence: Sequence[EvidenceHit],
    *,
    code_file_count: int = 0,
) -> list[Finding]:
    """Turn raw signals into findings."""
    real = tuple(hit for hit in evidence if not hit.is_test)
    test_only = tuple(hit for hit in evidence if hit.is_test)
    findings: list[Finding] = []

    # A document that also states, in so many words, that it does not estimate
    # depth is not making a false capability claim: it is contradicting itself.
    # That is reported below as a medium finding instead, which keeps the high
    # severity reserved for documentation a reader would actually be misled by.
    if claims and not evidence and not denials:
        findings.append(
            Finding(
                id="documented-depth-estimation-claim-false",
                severity="high",
                title="Documented depth-estimation claim is not backed by code",
                detail=(
                    f"{len(claims)} documentation sentence(s) advertise depth estimation, but "
                    f"no depth-estimation evidence was found in {code_file_count} scanned "
                    "Python file(s)."
                ),
                claims=tuple(claims),
            )
        )
    elif claims and not real and test_only and not denials:
        findings.append(
            Finding(
                id="documented-depth-estimation-claim-test-only",
                severity="medium",
                title="Depth estimation only appears in test code",
                detail=(
                    f"{len(claims)} documentation sentence(s) advertise depth estimation, but "
                    f"all {len(test_only)} evidence site(s) live in test files, so the shipped "
                    "code does not perform it."
                ),
                claims=tuple(claims),
                evidence=test_only,
            )
        )

    if denials and real:
        findings.append(
            Finding(
                id="denied-depth-estimation-contradicted",
                severity="high",
                title="Documentation denies depth estimation that the code performs",
                detail=(
                    f"{len(denials)} documentation sentence(s) state that depth estimation is "
                    f"not performed, but {len(real)} evidence site(s) perform it."
                ),
                denials=tuple(denials),
                evidence=real,
            )
        )

    if claims and denials:
        findings.append(
            Finding(
                id="contradictory-depth-estimation-documentation",
                severity="medium",
                title="Documentation both claims and denies depth estimation",
                detail=(
                    f"{len(claims)} sentence(s) advertise depth estimation while "
                    f"{len(denials)} sentence(s) deny it; readers cannot tell which is true."
                ),
                claims=tuple(claims),
                denials=tuple(denials),
            )
        )

    if real and not claims and not denials:
        findings.append(
            Finding(
                id="undocumented-depth-estimation",
                severity="low",
                title="Code performs depth estimation that documentation never mentions",
                detail=(
                    f"{len(real)} evidence site(s) perform depth estimation, but no scanned "
                    "documentation mentions it."
                ),
                evidence=real,
            )
        )

    order = {finding_id: index for index, finding_id in enumerate(FINDING_IDS)}
    findings.sort(key=lambda finding: (-SEVERITY_RANK[finding.severity], order[finding.id]))
    return findings


def audit_texts(
    docs: Sequence[tuple[str, str]] = (),
    code: Sequence[tuple[str, str]] = (),
) -> AuditResult:
    """Audit in-memory ``(path, text)`` pairs. Used by tests and by callers
    that already hold the file contents."""
    result = AuditResult()
    for path, text in docs:
        result.doc_files.append(path)
        claims, denials = scan_doc_text(text, path)
        result.claims.extend(claims)
        result.denials.extend(denials)
    for path, text in code:
        result.code_files.append(path)
        evidence, near_misses = scan_code_text(text, path)
        result.evidence.extend(evidence)
        result.near_misses.extend(near_misses)
    result.findings = build_findings(
        result.claims,
        result.denials,
        result.evidence,
        code_file_count=len(result.code_files),
    )
    return result


def audit_paths(
    paths: Iterable[os.PathLike[str] | str] = (),
    *,
    doc_paths: Iterable[os.PathLike[str] | str] = (),
    code_paths: Iterable[os.PathLike[str] | str] = (),
    exclude: Sequence[str] = (),
) -> AuditResult:
    """Audit files on disk.

    ``paths`` are classified by suffix; ``doc_paths`` and ``code_paths`` force
    the classification of the files they name.
    """
    paths = list(paths)
    doc_files = collect_doc_files(paths, exclude)
    code_files = collect_code_files(paths, exclude)
    doc_files += [
        path
        for path in collect_doc_files(list(doc_paths), exclude, force=True)
        if path not in doc_files
    ]
    code_files += [
        path
        for path in collect_code_files(list(code_paths), exclude, force=True)
        if path not in code_files
    ]
    return audit_texts(
        docs=[(path, _read_text(path)) for path in doc_files],
        code=[(path, _read_text(path)) for path in code_files],
    )
