"""Pattern tables for recognising real image depth-estimation work in Python source.

Nothing in this module imports, executes, or network-fetches the audited project.
Every decision is made from literal text: regexes over comment-stripped,
string-masked source, plus exact prose phrases for documentation.

The tables are deliberately conservative. When a signal is ambiguous the pattern
is left out, so the audit prefers a missed detection over a false accusation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

__all__ = [
    "CLAIM_PATTERNS",
    "CODE_SUFFIXES",
    "DENIAL_PATTERNS",
    "DOC_SUFFIXES",
    "GENERIC_DEPTH_CALLS",
    "INVOCATION_SPECS",
    "LOADER_PATTERNS",
    "MODEL_INVOCATION_ATTRS",
    "NON_DEPTH_CALL_NAMES",
    "NON_DEPTH_PIPELINE_TASKS",
    "NON_INVOCATION_ATTRS",
    "SKIP_DIRS",
    "InvocationSpec",
    "LoaderPattern",
    "has_chained_call",
    "iter_generic_calls",
    "iter_invocations",
    "iter_loader_matches",
    "line_of",
    "mask_string_literals",
    "normalize_source",
    "strip_comments",
]


# --------------------------------------------------------------------------- #
# File selection
# --------------------------------------------------------------------------- #

DOC_SUFFIXES = frozenset({".md", ".markdown", ".mdown", ".rst", ".txt", ".adoc"})
CODE_SUFFIXES = frozenset({".py", ".pyi"})

SKIP_DIRS = frozenset(
    {
        ".bzr",
        ".eggs",
        ".git",
        ".hg",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "site-packages",
        "venv",
    }
)


# --------------------------------------------------------------------------- #
# Source normalisation
# --------------------------------------------------------------------------- #

_QUOTES = ('"""', "'''", '"', "'")


def strip_comments(source: str) -> str:
    """Replace ``#`` comments with spaces, leaving string literals intact.

    Length and newline positions are preserved so that byte offsets taken from
    the result still map onto line numbers of the original text.
    """
    out: list[str] = []
    i = 0
    n = len(source)
    quote: str | None = None
    while i < n:
        ch = source[i]
        if quote is not None:
            if ch == "\\" and i + 1 < n:
                out.append(source[i : i + 2])
                i += 2
                continue
            if source.startswith(quote, i):
                out.append(quote)
                i += len(quote)
                quote = None
                continue
            out.append(ch)
            i += 1
            continue
        if ch in "\"'":
            for candidate in _QUOTES:
                if source.startswith(candidate, i):
                    quote = candidate
                    out.append(candidate)
                    i += len(candidate)
                    break
            continue
        if ch == "#":
            end = source.find("\n", i)
            if end == -1:
                end = n
            out.append(" " * (end - i))
            i = end
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def mask_string_literals(source: str) -> str:
    """Blank out the *contents* of string literals, keeping the quote marks.

    Newlines inside triple-quoted strings are kept, so line numbers are stable.
    """
    out: list[str] = []
    i = 0
    n = len(source)
    quote: str | None = None
    while i < n:
        ch = source[i]
        if quote is not None:
            if ch == "\\" and i + 1 < n:
                nxt = source[i + 1]
                out.append("\n" if nxt == "\n" else "  ")
                i += 2
                continue
            if source.startswith(quote, i):
                out.append(quote)
                i += len(quote)
                quote = None
                continue
            out.append("\n" if ch == "\n" else " ")
            i += 1
            continue
        if ch in "\"'":
            for candidate in _QUOTES:
                if source.startswith(candidate, i):
                    quote = candidate
                    out.append(candidate)
                    i += len(candidate)
                    break
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def normalize_source(source: str) -> str:
    """Comment-stripped and string-masked source, used for call detection."""
    return mask_string_literals(strip_comments(source))


def line_of(text: str, position: int) -> int:
    """1-based line number of ``position`` inside ``text``."""
    return text.count("\n", 0, position) + 1


# --------------------------------------------------------------------------- #
# Loader patterns: "a depth model / matcher was constructed here"
# --------------------------------------------------------------------------- #

# Optional assignment prefix. Rejects `==` comparisons and keyword arguments by
# refusing an assignment target that is preceded by `(`, `,`, `=`, a word
# character or a dot.
_ASSIGN = r"(?:(?<![\w.(,=])(?P<var>[A-Za-z_][\w.]*)\s*=(?!=)\s*)?"


@dataclass(frozen=True)
class LoaderPattern:
    """A literal construction site for a depth estimator."""

    name: str
    backend: str
    invocation: str
    regex: re.Pattern[str]
    description: str
    #: Label used when the regex has no ``symbol`` group to name the construct.
    symbol: str = ""


@dataclass(frozen=True)
class InvocationSpec:
    """How a loaded object has to be used before it counts as evidence."""

    allow_bare: bool
    attrs: tuple[str, ...]


MODEL_INVOCATION_ATTRS: tuple[str, ...] = (
    "estimate",
    "estimate_depth",
    "forward",
    "infer",
    "infer_image",
    "infer_pil",
    "predict",
    "predict_depth",
    "run",
)

PIPELINE_INVOCATION_ATTRS: tuple[str, ...] = ("predict",)

STEREO_INVOCATION_ATTRS: tuple[str, ...] = ("compute", "compute_disparity")

#: Attributes that are explicitly *not* an inference call. Loading a model and
#: only touching one of these is a near miss, never evidence.
NON_INVOCATION_ATTRS: tuple[str, ...] = (
    "apply",
    "children",
    "compile",
    "cpu",
    "cuda",
    "double",
    "eval",
    "fit",
    "float",
    "half",
    "load_state_dict",
    "modules",
    "named_parameters",
    "parameters",
    "requires_grad_",
    "save_pretrained",
    "share_memory",
    "state_dict",
    "to",
    "train",
    "zero_grad",
)

INVOCATION_SPECS: dict[str, InvocationSpec] = {
    "model": InvocationSpec(allow_bare=True, attrs=MODEL_INVOCATION_ATTRS),
    "pipeline": InvocationSpec(allow_bare=True, attrs=PIPELINE_INVOCATION_ATTRS),
    "stereo": InvocationSpec(allow_bare=False, attrs=STEREO_INVOCATION_ATTRS),
}


LOADER_PATTERNS: tuple[LoaderPattern, ...] = (
    LoaderPattern(
        name="hf-pipeline-depth-estimation",
        backend="transformers",
        invocation="pipeline",
        regex=re.compile(
            _ASSIGN
            + r"(?:transformers\.)?pipeline\s*\(\s*(?:[^()]*?,\s*)?"
            r"(?:task\s*=\s*)?(?P<q>['\"])depth[-_]estimation(?P=q)"
        ),
        description='transformers pipeline("depth-estimation")',
        symbol='pipeline("depth-estimation")',
    ),
    LoaderPattern(
        name="transformers-depth-estimation-model",
        backend="transformers",
        invocation="model",
        regex=re.compile(
            _ASSIGN + r"(?:transformers\.)?(?P<symbol>[A-Za-z_]\w*ForDepthEstimation)"
            r"\s*(?:\.from_pretrained|\.from_config)?\s*\("
        ),
        description="transformers *ForDepthEstimation model class "
        "(AutoModelForDepthEstimation, DPTForDepthEstimation, ...)",
    ),
    LoaderPattern(
        name="depth-anything-model",
        backend="depth-anything",
        invocation="model",
        regex=re.compile(
            _ASSIGN + r"(?P<symbol>DepthAnything\w*)"
            r"\s*(?:\.from_pretrained|\.from_config)?\s*\("
        ),
        description="Depth Anything model class",
    ),
    LoaderPattern(
        name="torch-hub-midas",
        backend="midas",
        invocation="model",
        regex=re.compile(
            _ASSIGN + r"torch\.hub\.load\s*\(\s*(?:repo_or_dir\s*=\s*)?"
            r"(?P<q>['\"])(?:intel-isl|isl-org)/MiDaS(?P=q)"
        ),
        description='torch.hub.load("intel-isl/MiDaS", ...)',
        symbol='torch.hub.load("intel-isl/MiDaS")',
    ),
    LoaderPattern(
        name="torch-hub-zoedepth",
        backend="zoedepth",
        invocation="model",
        regex=re.compile(
            _ASSIGN + r"torch\.hub\.load\s*\(\s*(?:repo_or_dir\s*=\s*)?"
            r"(?P<q>['\"])(?:isl-org|intel-isl)/ZoeDepth(?P=q)"
        ),
        description='torch.hub.load("isl-org/ZoeDepth", ...)',
        symbol='torch.hub.load("isl-org/ZoeDepth")',
    ),
    LoaderPattern(
        name="midas-model-class",
        backend="midas",
        invocation="model",
        regex=re.compile(
            _ASSIGN + r"(?P<symbol>MidasNet_small|MidasNet|MiDaSNet|DPTDepthModel|MiDaS)\s*\("
        ),
        description="MiDaS network class instantiation",
    ),
    LoaderPattern(
        name="zoedepth-model-class",
        backend="zoedepth",
        invocation="model",
        regex=re.compile(
            _ASSIGN + r"(?P<symbol>ZoeDepth\w*|ZoeD_NK|ZoeD_N|ZoeD_K)\s*\("
        ),
        description="ZoeDepth network class instantiation",
    ),
    LoaderPattern(
        name="opencv-stereo-matcher",
        backend="opencv",
        invocation="stereo",
        regex=re.compile(
            _ASSIGN + r"(?:cv2?\.)?(?P<symbol>StereoSGBM_create|StereoBM_create"
            r"|StereoSGBM\.create|StereoBM\.create)\s*\("
        ),
        description="cv2 stereo matcher construction",
    ),
)


def iter_loader_matches(code: str) -> Iterator[tuple[LoaderPattern, re.Match[str]]]:
    """Yield loader matches over comment-stripped source, left to right.

    Overlapping matches from different patterns at the same offset are reported
    once only, using the earliest pattern in :data:`LOADER_PATTERNS`.
    """
    seen: set[int] = set()
    hits: list[tuple[int, int, LoaderPattern, re.Match[str]]] = []
    for order, pattern in enumerate(LOADER_PATTERNS):
        for match in pattern.regex.finditer(code):
            hits.append((match.start(), order, pattern, match))
    for start, _order, pattern, match in sorted(hits, key=lambda item: (item[0], item[1])):
        anchor = match.end()
        if anchor in seen:
            continue
        seen.add(anchor)
        yield pattern, match


def iter_invocations(
    masked: str, var: str, spec: InvocationSpec
) -> Iterator[tuple[int, str]]:
    """Yield ``(position, attribute)`` for every call made on ``var``.

    ``attribute`` is ``"__call__"`` for a bare ``var(...)`` call.
    """
    escaped = re.escape(var)
    probes: list[tuple[re.Pattern[str], str]] = []
    if spec.allow_bare:
        probes.append((re.compile(r"(?<![\w.])" + escaped + r"\s*\("), "__call__"))
    for attr in spec.attrs:
        probes.append(
            (
                re.compile(r"(?<![\w.])" + escaped + r"\s*\.\s*" + re.escape(attr) + r"\s*\("),
                attr,
            )
        )
    found: list[tuple[int, str]] = []
    for regex, attr in probes:
        for match in regex.finditer(masked):
            found.append((match.start(), attr))
    yield from sorted(found)


_CHAINED_BARE_RE = re.compile(r"\)\s*\(")


def has_chained_call(masked_line: str, spec: InvocationSpec) -> str | None:
    """Detect ``loader(...)(x)`` / ``loader(...).compute(x)`` on one line.

    ``masked_line`` is the remainder of the (string-masked) line that starts at
    the loader match. Returns the invoked attribute name, or ``None``.
    """
    for attr in spec.attrs:
        if re.search(r"\)\s*\.\s*" + re.escape(attr) + r"\s*\(", masked_line):
            return attr
    if spec.allow_bare and _CHAINED_BARE_RE.search(masked_line):
        return "__call__"
    return None


# --------------------------------------------------------------------------- #
# Generic depth-named calls
# --------------------------------------------------------------------------- #

#: Function names that are specific enough to count as depth-estimation work on
#: their own, provided they are actually *called* (not merely defined).
GENERIC_DEPTH_CALLS: tuple[str, ...] = (
    "compute_depth",
    "compute_depth_map",
    "compute_disparity",
    "depth_estimation",
    "depth_from_image",
    "depth_inference",
    "depth_map",
    "depth_maps",
    "disparity_map",
    "estimate_depth",
    "estimate_depth_map",
    "estimate_disparity",
    "extract_depth_map",
    "generate_depth_map",
    "get_depth_map",
    "infer_depth",
    "infer_depth_map",
    "monocular_depth",
    "monocular_depth_estimation",
    "predict_depth",
    "predict_depth_map",
    "predict_disparity",
    "run_depth_estimation",
    "run_depth_inference",
)

_GENERIC_CALL_RE = re.compile(
    r"(?<![\w.])(?:(?P<owner>[A-Za-z_][\w.]*)\.)?(?P<name>"
    + "|".join(sorted(GENERIC_DEPTH_CALLS, key=len, reverse=True))
    + r")\s*\("
)

_DEFINITION_RE = re.compile(r"\b(?:def|class)\s+$")


def iter_generic_calls(masked: str) -> Iterator[tuple[int, str, str | None]]:
    """Yield ``(position, name, owner)`` for generic depth calls in ``masked``.

    ``def``/``class`` statements are skipped: declaring ``estimate_depth`` is
    not the same as running it.
    """
    for match in _GENERIC_CALL_RE.finditer(masked):
        if _DEFINITION_RE.search(masked[: match.start()]):
            continue
        yield match.start(), match.group("name"), match.group("owner")


# --------------------------------------------------------------------------- #
# Counter-examples (documentation + test material, never evidence)
# --------------------------------------------------------------------------- #

#: transformers pipeline tasks that are *not* depth estimation.
NON_DEPTH_PIPELINE_TASKS: tuple[str, ...] = (
    "feature-extraction",
    "image-classification",
    "image-segmentation",
    "image-to-text",
    "instance-segmentation",
    "keypoint-detection",
    "mask-generation",
    "object-detection",
    "optical-flow",
    "panoptic-segmentation",
    "pose-estimation",
    "semantic-segmentation",
    "video-classification",
    "zero-shot-object-detection",
)

#: Call names that must never be read as depth estimation: neighbouring vision
#: tasks plus words where "depth" means something else entirely.
NON_DEPTH_CALL_NAMES: tuple[str, ...] = (
    "bit_depth",
    "classify_image",
    "color_depth",
    "compute_histogram",
    "compute_optical_flow",
    "compute_pose",
    "depth_first_search",
    "depth_limit",
    "detect_faces",
    "detect_keypoints",
    "detect_objects",
    "estimate_flow",
    "estimate_homography",
    "estimate_pose",
    "get_tree_depth",
    "max_depth",
    "predict_flow",
    "predict_pose",
    "recursion_depth",
    "search_depth",
    "segment_image",
    "set_max_depth",
    "stack_depth",
    "track_points",
)


# --------------------------------------------------------------------------- #
# Documentation phrases
# --------------------------------------------------------------------------- #

_DEPTH_NOUN = r"(?:per-pixel\s+|per\s+pixel\s+|dense\s+|metric\s+|relative\s+|monocular\s+|scene\s+)*depth"

CLAIM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in (
        ("depth-estimation", r"\bdepth[\s\-]?estimation\b"),
        ("depth-estimator", r"\bdepth\s+estimators?\b"),
        ("depth-map", r"\bdepth[\s\-]maps?\b"),
        ("depth-prediction", r"\bdepth\s+predictions?\b"),
        ("depth-inference", r"\bdepth\s+inference\b"),
        ("disparity-map", r"\bdisparity[\s\-]maps?\b"),
        ("monocular-depth", r"\bmonocular\s+depth\b"),
        ("metric-depth", r"\bmetric\s+depth\b"),
        ("relative-depth", r"\brelative\s+depth\b"),
        ("stereo-depth", r"\bstereo\s+depth\b"),
        (
            "estimates-depth",
            r"\b(?:estimate|estimates|estimated|estimating)\s+(?:the\s+|a\s+)?" + _DEPTH_NOUN + r"\b",
        ),
        (
            "predicts-depth",
            r"\b(?:predict|predicts|predicted|predicting)\s+(?:the\s+|a\s+)?" + _DEPTH_NOUN + r"\b",
        ),
        (
            "computes-depth",
            r"\b(?:compute|computes|computed|computing)\s+(?:the\s+|a\s+)?" + _DEPTH_NOUN + r"\b",
        ),
        (
            "infers-depth",
            r"\b(?:infer|infers|inferred|inferring)\s+(?:the\s+|a\s+)?" + _DEPTH_NOUN + r"\b",
        ),
        ("depth-from-single-image", r"\bdepth\s+from\s+(?:a\s+)?single\s+(?:rgb\s+)?image\b"),
        ("midas", r"\bmidas\b"),
        ("depth-anything", r"\bdepth[\s\-]?anything\b"),
        ("zoedepth", r"\bzoe[\s\-]?depth\b"),
    )
)

DENIAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in (
        ("no-depth-estimation", r"\bno\s+(?:monocular\s+)?depth[\s\-]?estimation\b"),
        ("no-depth-maps", r"\bno\s+depth[\s\-]maps?\b"),
        ("no-disparity-maps", r"\bno\s+disparity[\s\-]maps?\b"),
        (
            "does-not-perform-depth-estimation",
            r"\bdoes\s+not\s+(?:perform|do|run|support|provide|include|implement|attempt|ship)\s+"
            r"(?:any\s+)?(?:monocular\s+)?depth[\s\-]?estimation\b",
        ),
        (
            "does-not-estimate-depth",
            r"\bdoes\s+not\s+(?:estimate|predict|compute|infer|produce|output|generate|return)\s+"
            r"(?:any\s+)?(?:depth|depth[\s\-]maps?|disparity[\s\-]maps?)\b",
        ),
        ("without-depth-estimation", r"\bwithout\s+(?:any\s+)?depth[\s\-]?estimation\b"),
        ("not-a-depth-estimator", r"\b(?:is\s+)?not\s+a\s+depth[\s\-]?(?:estimation|estimator)\b"),
        (
            "never-estimates-depth",
            r"\bnever\s+(?:estimates|predicts|computes|infers)\s+(?:any\s+)?depth\b",
        ),
        (
            "depth-estimation-is-not-supported",
            r"\bdepth[\s\-]?estimation\s+is\s+not\s+"
            r"(?:performed|supported|implemented|included|provided|done)\b",
        ),
        (
            "cannot-estimate-depth",
            r"\b(?:cannot|can\s?not|can't)\s+(?:estimate|predict|compute|infer)\s+depth\b",
        ),
    )
)
