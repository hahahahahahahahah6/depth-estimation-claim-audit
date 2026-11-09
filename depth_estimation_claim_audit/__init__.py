"""depth-estimation-claim-audit: offline audit of documented depth-estimation claims.

Compare what a project's documentation says about image depth estimation with
what its Python sources actually do. Everything is static text analysis: no
imports of the audited code, no model downloads, no network access.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .audit import (
    FAIL_ON_CHOICES,
    FINDING_IDS,
    SEVERITIES,
    SEVERITY_RANK,
    AuditError,
    AuditResult,
    ClaimHit,
    DenialHit,
    EvidenceHit,
    Finding,
    NearMiss,
    audit_paths,
    audit_texts,
    build_findings,
    is_test_path,
    scan_code_text,
    scan_doc_text,
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
    "__version__",
    "audit_paths",
    "audit_texts",
    "build_findings",
    "is_test_path",
    "scan_code_text",
    "scan_doc_text",
]
