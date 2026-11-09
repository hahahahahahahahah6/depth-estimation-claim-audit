"""Unit tests for depth-estimation-claim-audit."""

from __future__ import annotations

import pytest

import depth_estimation_claim_audit as pkg
from depth_estimation_claim_audit import evidence as ev
from depth_estimation_claim_audit.audit import (
    FAIL_ON_CHOICES,
    FINDING_IDS,
    SEVERITIES,
    SEVERITY_RANK,
    AuditError,
    Finding,
    audit_paths,
    audit_texts,
    build_findings,
    collect_code_files,
    collect_doc_files,
    doc_segments,
    exit_code_for,
    is_test_path,
    normalize_path,
    scan_code_text,
    scan_doc_text,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def evidence_of(source: str, path: str = "pkg/mod.py"):
    hits, _ = scan_code_text(source, path)
    return hits


def near_misses_of(source: str, path: str = "pkg/mod.py"):
    _, misses = scan_code_text(source, path)
    return misses


def claim_phrases(text: str) -> list[str]:
    claims, _ = scan_doc_text(text, "README.md")
    return [hit.phrase for hit in claims]


def denial_phrases(text: str) -> list[str]:
    _, denials = scan_doc_text(text, "README.md")
    return [hit.phrase for hit in denials]


# --------------------------------------------------------------------------- #
# Package metadata
# --------------------------------------------------------------------------- #


def test_version_is_declared():
    assert pkg.__version__ == "0.1.0"


def test_public_api_is_importable():
    for name in ("audit_paths", "audit_texts", "scan_code_text", "scan_doc_text"):
        assert callable(getattr(pkg, name))


def test_severity_scale():
    assert SEVERITIES == ("low", "medium", "high")
    assert SEVERITY_RANK == {"low": 1, "medium": 2, "high": 3}


def test_fail_on_choices():
    assert FAIL_ON_CHOICES == ("none", "low", "medium", "high")


def test_finding_ids_are_unique():
    assert len(set(FINDING_IDS)) == len(FINDING_IDS)


def test_generic_depth_call_table_size():
    assert len(ev.GENERIC_DEPTH_CALLS) == 24
    assert len(set(ev.GENERIC_DEPTH_CALLS)) == 24


def test_loader_pattern_names_are_unique():
    names = [pattern.name for pattern in ev.LOADER_PATTERNS]
    assert len(set(names)) == len(names)


def test_every_loader_pattern_has_an_invocation_spec():
    for pattern in ev.LOADER_PATTERNS:
        assert pattern.invocation in ev.INVOCATION_SPECS


def test_invocation_and_non_invocation_attrs_do_not_overlap():
    invoking = set(ev.MODEL_INVOCATION_ATTRS) | set(ev.STEREO_INVOCATION_ATTRS)
    assert invoking.isdisjoint(ev.NON_INVOCATION_ATTRS)


# --------------------------------------------------------------------------- #
# Source normalisation
# --------------------------------------------------------------------------- #


def test_strip_comments_removes_comment_text():
    assert "secret" not in ev.strip_comments("x = 1  # secret\n")


def test_strip_comments_preserves_length():
    source = 'a = 1  # note\nb = "# not a comment"\n'
    assert len(ev.strip_comments(source)) == len(source)


def test_strip_comments_keeps_string_contents():
    assert "depth-estimation" in ev.strip_comments('pipeline("depth-estimation")')


def test_strip_comments_ignores_hash_inside_string():
    stripped = ev.strip_comments('url = "http://x#y"  # trailing\n')
    assert "http://x#y" in stripped
    assert "trailing" not in stripped


def test_mask_string_literals_blanks_contents():
    masked = ev.mask_string_literals('s = "estimate_depth(x)"\n')
    assert "estimate_depth" not in masked
    assert masked.count('"') == 2


def test_mask_string_literals_preserves_length_and_lines():
    source = 'x = """a\nb\nc"""\ny = 2\n'
    masked = ev.mask_string_literals(source)
    assert len(masked) == len(source)
    assert masked.count("\n") == source.count("\n")


def test_normalize_source_handles_both():
    masked = ev.normalize_source('x = "estimate_depth(a)"  # estimate_depth(b)\n')
    assert "estimate_depth" not in masked


def test_line_of_counts_from_one():
    text = "a\nb\nc"
    assert ev.line_of(text, 0) == 1
    assert ev.line_of(text, 2) == 2
    assert ev.line_of(text, 4) == 3


# --------------------------------------------------------------------------- #
# Path classification
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "path",
    [
        "tests/test_audit.py",
        "tests/helpers.py",
        "pkg/tests/thing.py",
        "test_depth.py",
        "depth_test.py",
        "conftest.py",
        "src/testing/fixture.py",
        "a/b/_tests/c.py",
    ],
)
def test_is_test_path_true(path):
    assert is_test_path(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "pkg/depth.py",
        "src/latest_model.py",
        "contest.py",
        "pkg/attestation.py",
        "pipeline/run.py",
        "README.md",
    ],
)
def test_is_test_path_false(path):
    assert is_test_path(path) is False


def test_normalize_path_uses_forward_slashes():
    assert normalize_path("a\\b\\c.py") == "a/b/c.py"


# --------------------------------------------------------------------------- #
# Loader + invocation evidence (positive)
# --------------------------------------------------------------------------- #

POSITIVE_SOURCES = {
    "pipeline-positional": (
        "hf-pipeline-depth-estimation",
        'from transformers import pipeline\n'
        'pipe = pipeline("depth-estimation")\n'
        "out = pipe(image)\n",
    ),
    "pipeline-single-quotes": (
        "hf-pipeline-depth-estimation",
        "pipe = pipeline('depth-estimation')\nout = pipe(image)\n",
    ),
    "pipeline-task-kwarg": (
        "hf-pipeline-depth-estimation",
        'pipe = pipeline(task="depth-estimation")\nout = pipe(img)\n',
    ),
    "pipeline-model-first": (
        "hf-pipeline-depth-estimation",
        'pipe = pipeline(model="Intel/dpt-large", task="depth-estimation")\nd = pipe(img)\n',
    ),
    "pipeline-module-qualified": (
        "hf-pipeline-depth-estimation",
        'p = transformers.pipeline("depth-estimation")\nresult = p(frame)\n',
    ),
    "pipeline-underscore-task": (
        "hf-pipeline-depth-estimation",
        'p = pipeline("depth_estimation")\nresult = p(frame)\n',
    ),
    "pipeline-multiline": (
        "hf-pipeline-depth-estimation",
        'estimator = pipeline(\n    task="depth-estimation",\n    device=0,\n)\n'
        "depth = estimator(image)\n",
    ),
    "pipeline-chained-call": (
        "hf-pipeline-depth-estimation",
        'depth = pipeline("depth-estimation")(image)\n',
    ),
    "pipeline-predict-attr": (
        "hf-pipeline-depth-estimation",
        'p = pipeline("depth-estimation")\nout = p.predict(image)\n',
    ),
    "pipeline-attribute-target": (
        "hf-pipeline-depth-estimation",
        'self.pipe = pipeline("depth-estimation")\nout = self.pipe(image)\n',
    ),
    "automodel-from-pretrained": (
        "transformers-depth-estimation-model",
        'model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")\n'
        "with torch.no_grad():\n"
        "    outputs = model(**inputs)\n",
    ),
    "automodel-module-qualified": (
        "transformers-depth-estimation-model",
        'm = transformers.AutoModelForDepthEstimation.from_pretrained("x")\ny = m(**inputs)\n',
    ),
    "dpt-for-depth-estimation": (
        "transformers-depth-estimation-model",
        'model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")\n'
        "pred = model(pixel_values=pv)\n",
    ),
    "glpn-for-depth-estimation": (
        "transformers-depth-estimation-model",
        'model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")\n'
        "pred = model(pixel_values=pv)\n",
    ),
    "depth-anything-for-depth-estimation": (
        "transformers-depth-estimation-model",
        'model = DepthAnythingForDepthEstimation.from_pretrained("x")\nout = model(**inputs)\n',
    ),
    "zoedepth-for-depth-estimation": (
        "transformers-depth-estimation-model",
        'model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu")\n'
        "out = model(**inputs)\n",
    ),
    "depth-anything-class": (
        "depth-anything-model",
        'model = DepthAnything.from_pretrained("LiheYoung/depth_anything_vitl14")\n'
        "depth = model(image)\n",
    ),
    "depth-anything-v2-class": (
        "depth-anything-model",
        "model = DepthAnythingV2(encoder='vitl')\ndepth = model.infer_image(raw)\n",
    ),
    "midas-torch-hub": (
        "torch-hub-midas",
        'midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")\n'
        "midas.to(device)\n"
        "midas.eval()\n"
        "prediction = midas(input_batch)\n",
    ),
    "midas-torch-hub-new-org": (
        "torch-hub-midas",
        'midas = torch.hub.load("isl-org/MiDaS", "MiDaS_small")\nout = midas(batch)\n',
    ),
    "midas-torch-hub-kwarg-repo": (
        "torch-hub-midas",
        'midas = torch.hub.load(repo_or_dir="intel-isl/MiDaS", model="DPT_Hybrid")\n'
        "out = midas.forward(batch)\n",
    ),
    "midas-network-class": (
        "midas-model-class",
        'model = DPTDepthModel(path="weights.pt", backbone="vitl16_384")\nout = model(sample)\n',
    ),
    "midas-net-small": (
        "midas-model-class",
        "model = MidasNet_small(path=None)\nout = model(sample)\n",
    ),
    "midas-net": (
        "midas-model-class",
        "net = MidasNet(weights)\nprediction = net(batch)\n",
    ),
    "zoedepth-torch-hub": (
        "torch-hub-zoedepth",
        'zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True)\n'
        "depth = zoe.infer_pil(image)\n",
    ),
    "zoedepth-torch-hub-bare-call": (
        "torch-hub-zoedepth",
        'zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK")\ndepth = zoe(tensor)\n',
    ),
    "zoedepth-class": (
        "zoedepth-model-class",
        "model = ZoeDepth(config)\ndepth = model.infer(image)\n",
    ),
    "zoedepth-build-class": (
        "zoedepth-model-class",
        "model = ZoeD_N(config)\ndepth = model.predict(image)\n",
    ),
    "stereo-sgbm": (
        "opencv-stereo-matcher",
        "stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=15)\n"
        "disparity = stereo.compute(left, right)\n",
    ),
    "stereo-bm": (
        "opencv-stereo-matcher",
        "matcher = cv2.StereoBM_create(numDisparities=16, blockSize=15)\n"
        "disp = matcher.compute(gray_l, gray_r)\n",
    ),
    "stereo-bm-dot-create": (
        "opencv-stereo-matcher",
        "matcher = cv2.StereoBM.create(16, 15)\ndisp = matcher.compute(gray_l, gray_r)\n",
    ),
    "stereo-imported-name": (
        "opencv-stereo-matcher",
        "from cv2 import StereoSGBM_create\n"
        "stereo = StereoSGBM_create(0, 64, 3)\n"
        "disp = stereo.compute(l, r)\n",
    ),
    "stereo-attribute-target": (
        "opencv-stereo-matcher",
        "self.stereo = cv2.StereoSGBM_create(0, 64)\n"
        "disp = self.stereo.compute(l, r)\n",
    ),
    "stereo-chained": (
        "opencv-stereo-matcher",
        "disp = cv2.StereoBM_create(16, 15).compute(left, right)\n",
    ),
    "call-before-load-in-file": (
        "torch-hub-midas",
        "def run(batch):\n"
        "    return midas(batch)\n"
        "\n"
        'midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")\n',
    ),
}


@pytest.mark.parametrize(
    ("case", "pattern", "source"),
    [(case, pattern, source) for case, (pattern, source) in POSITIVE_SOURCES.items()],
    ids=list(POSITIVE_SOURCES),
)
def test_real_depth_estimation_is_evidence(case, pattern, source):
    hits = evidence_of(source)
    assert hits, f"{case}: expected evidence"
    assert pattern in {hit.pattern for hit in hits}


@pytest.mark.parametrize(
    ("case", "pattern", "source"),
    [(case, pattern, source) for case, (pattern, source) in POSITIVE_SOURCES.items()],
    ids=list(POSITIVE_SOURCES),
)
def test_positive_sources_report_no_near_miss(case, pattern, source):
    assert near_misses_of(source) == []


def test_evidence_records_loader_and_call_lines():
    source = (
        'midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")\n'
        "midas.eval()\n"
        "prediction = midas(batch)\n"
    )
    (hit,) = evidence_of(source)
    assert hit.loader_line == 1
    assert hit.line == 3
    assert hit.invocation == "__call__"
    assert hit.backend == "midas"
    assert hit.kind == "loader-and-call"
    assert hit.snippet == "prediction = midas(batch)"


def test_evidence_marks_test_files():
    source = 'p = pipeline("depth-estimation")\nout = p(img)\n'
    (hit,) = evidence_of(source, "tests/test_depth.py")
    assert hit.is_test is True


def test_evidence_in_shipped_code_is_not_test():
    source = 'p = pipeline("depth-estimation")\nout = p(img)\n'
    (hit,) = evidence_of(source, "pkg/depth.py")
    assert hit.is_test is False


def test_evidence_is_deduplicated_per_site():
    source = 'p = pipeline("depth-estimation")\nout = p(img)\nout2 = p(img2)\n'
    hits = evidence_of(source)
    assert len(hits) == 1


# --------------------------------------------------------------------------- #
# Loader + invocation evidence (negative)
# --------------------------------------------------------------------------- #

NEGATIVE_SOURCES = {
    "pipeline-never-called": 'p = pipeline("depth-estimation")\n',
    "pipeline-only-trained": 'p = pipeline("depth-estimation")\np.train()\n',
    "pipeline-wrong-task": 'p = pipeline("image-classification")\nout = p(img)\n',
    "pipeline-task-in-variable": "p = pipeline(task)\nout = p(img)\n",
    "pipeline-task-from-config": 'p = pipeline(cfg["task"])\nout = p(img)\n',
    "automodel-only-loaded": 'm = AutoModelForDepthEstimation.from_pretrained("x")\n',
    "automodel-only-moved": (
        'm = AutoModelForDepthEstimation.from_pretrained("x")\nm.to(device)\nm.eval()\n'
    ),
    "automodel-only-trained": (
        'm = AutoModelForDepthEstimation.from_pretrained("x")\nm.train()\n'
        "m.load_state_dict(sd)\n"
    ),
    "automodel-only-saved": (
        'm = AutoModelForDepthEstimation.from_pretrained("x")\nm.save_pretrained("out")\n'
    ),
    "automodel-params-only": (
        'm = AutoModelForDepthEstimation.from_pretrained("x")\nopt = SGD(m.parameters())\n'
    ),
    "midas-only-loaded": 'midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")\n',
    "midas-only-trained": (
        'midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")\nmidas.train()\n'
    ),
    "midas-repo-in-variable": (
        'repo = "intel-isl/MiDaS"\nm = torch.hub.load(repo, "DPT_Large")\nout = m(x)\n'
    ),
    "midas-repo-formatted": (
        'm = torch.hub.load(f"{org}/MiDaS", "DPT_Large")\nout = m(x)\n'
    ),
    "torch-hub-other-repo": (
        'm = torch.hub.load("pytorch/vision", "resnet18")\nout = m(x)\n'
    ),
    "zoedepth-only-loaded": 'zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N")\n',
    "zoedepth-repo-in-variable": (
        'm = torch.hub.load(ZOE_REPO, "ZoeD_N")\ndepth = m.infer_pil(img)\n'
    ),
    "depth-anything-import-only": "from depth_anything.dpt import DepthAnything\n",
    "depth-anything-only-loaded": 'm = DepthAnything.from_pretrained("x")\nm.eval()\n',
    "stereo-never-computed": "stereo = cv2.StereoBM_create(16, 15)\n",
    "stereo-only-configured": (
        "stereo = cv2.StereoSGBM_create(0, 64)\nstereo.setBlockSize(5)\n"
    ),
    "stereo-bare-call-not-compute": (
        "stereo = cv2.StereoSGBM_create(0, 64)\ndisp = stereo(l, r)\n"
    ),
    "object-detection-pipeline": 'p = pipeline("object-detection")\nout = p(img)\n',
    "segmentation-pipeline": 'p = pipeline("image-segmentation")\nout = p(img)\n',
    "yolo-detection": 'model = YOLO("yolov8n.pt")\nresults = model(frame)\n',
    "optical-flow-only": (
        "flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)\n"
    ),
    "pose-estimation-only": "keypoints = estimate_pose(frame)\n",
    "classification-only": "label = classify_image(frame)\n",
    "evidence-in-comment": '# p = pipeline("depth-estimation")\n# out = p(img)\n',
    "evidence-in-docstring": (
        '"""Example:\n\np = pipeline("depth-estimation")\nout = p(img)\n"""\n'
    ),
    "evidence-in-string-constant": (
        'SNIPPET = "midas = torch.hub.load(\'intel-isl/MiDaS\', \'DPT_Large\'); midas(x)"\n'
    ),
    "plain-tensor-ops": "import torch\nx = torch.zeros(3, 3)\ny = x.mean()\n",
    "empty-file": "",
    "comment-only-file": "# nothing here\n",
}


@pytest.mark.parametrize(
    ("case", "source"),
    list(NEGATIVE_SOURCES.items()),
    ids=list(NEGATIVE_SOURCES),
)
def test_non_depth_estimation_is_not_evidence(case, source):
    assert evidence_of(source) == [], case


@pytest.mark.parametrize(
    "task",
    ev.NON_DEPTH_PIPELINE_TASKS,
)
def test_other_pipeline_tasks_are_not_evidence(task):
    source = f'p = pipeline("{task}")\nout = p(img)\n'
    assert evidence_of(source) == []


# --------------------------------------------------------------------------- #
# Near misses
# --------------------------------------------------------------------------- #


def test_loader_without_invocation_is_a_near_miss():
    (miss,) = near_misses_of('p = pipeline("depth-estimation")\n')
    assert miss.reason == "loader-without-invocation"
    assert miss.pattern == "hf-pipeline-depth-estimation"
    assert miss.line == 1


def test_train_only_model_is_a_near_miss():
    (miss,) = near_misses_of(
        'midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")\nmidas.train()\n'
    )
    assert miss.pattern == "torch-hub-midas"


def test_near_miss_is_not_counted_as_evidence():
    result = audit_texts(code=[("pkg/a.py", 'p = pipeline("depth-estimation")\n')])
    assert result.evidence == []
    assert len(result.near_misses) == 1
    assert result.findings == []


def test_near_miss_snippet_is_recorded():
    (miss,) = near_misses_of("stereo = cv2.StereoBM_create(16, 15)\n")
    assert miss.snippet == "stereo = cv2.StereoBM_create(16, 15)"


# --------------------------------------------------------------------------- #
# Generic depth-named calls
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ev.GENERIC_DEPTH_CALLS)
def test_generic_depth_call_is_evidence(name):
    hits = evidence_of(f"value = {name}(frame)\n")
    assert [hit.pattern for hit in hits] == ["generic-depth-call"]
    assert hits[0].symbol == name
    assert hits[0].backend == "generic"


@pytest.mark.parametrize("name", ev.GENERIC_DEPTH_CALLS)
def test_generic_depth_definition_alone_is_not_evidence(name):
    assert evidence_of(f"def {name}(frame):\n    return None\n") == []


@pytest.mark.parametrize("name", ev.NON_DEPTH_CALL_NAMES)
def test_non_depth_call_names_are_not_evidence(name):
    assert evidence_of(f"value = {name}(frame)\n") == []


def test_generic_call_on_attribute_is_evidence():
    (hit,) = evidence_of("d = self.predict_depth(image)\n")
    assert hit.symbol == "self.predict_depth"
    assert hit.invocation == "predict_depth"


def test_generic_call_on_module_is_evidence():
    (hit,) = evidence_of("d = depthlib.estimate_depth(image)\n")
    assert hit.symbol == "depthlib.estimate_depth"


def test_generic_call_in_comment_is_not_evidence():
    assert evidence_of("# estimate_depth(image)\n") == []


def test_generic_call_in_string_is_not_evidence():
    assert evidence_of('doc = "call estimate_depth(image) first"\n') == []


def test_generic_name_without_call_is_not_evidence():
    assert evidence_of("handler = estimate_depth\n") == []


def test_generic_name_as_prefix_is_not_evidence():
    assert evidence_of("x = estimate_depth_of_field(image)\n") == []


def test_generic_name_with_word_prefix_is_not_evidence():
    assert evidence_of("x = fake_estimate_depth(image)\n") == []


def test_class_definition_is_not_a_generic_call():
    assert evidence_of("class depth_estimation(object):\n    pass\n") == []


def test_async_definition_is_not_a_generic_call():
    assert evidence_of("async def estimate_depth(x):\n    return 1\n") == []


def test_generic_call_line_number():
    source = "import os\n\n\nd = compute_depth(img)\n"
    (hit,) = evidence_of(source)
    assert hit.line == 4


def test_multiple_generic_calls_are_all_reported():
    source = "a = estimate_depth(x)\nb = predict_disparity(y)\n"
    hits = evidence_of(source)
    assert {hit.symbol for hit in hits} == {"estimate_depth", "predict_disparity"}


def test_definition_then_call_is_evidence():
    source = "def estimate_depth(img):\n    return midas(img)\n\nd = estimate_depth(frame)\n"
    hits = evidence_of(source)
    assert [hit.line for hit in hits] == [4]


# --------------------------------------------------------------------------- #
# Documentation claims
# --------------------------------------------------------------------------- #

CLAIM_DOCS = {
    "depth-estimation": "The tool runs depth estimation on every frame.",
    "depth-estimation-hyphen": "A depth-estimation stage follows detection.",
    "depth-estimator": "Ships a depth estimator for indoor scenes.",
    "depth-map": "Returns a depth map per image.",
    "depth-map-hyphen": "Returns a dense depth-map per image.",
    "depth-prediction": "Depth prediction happens on the GPU.",
    "depth-inference": "Depth inference takes 30 ms.",
    "disparity-map": "Computes a disparity map from the stereo pair.",
    "monocular-depth": "Monocular depth is recovered from one photo.",
    "metric-depth": "Reports metric depth in metres.",
    "relative-depth": "Reports relative depth only.",
    "stereo-depth": "Stereo depth is available for calibrated rigs.",
    "estimates-depth": "We estimate depth for each frame.",
    "estimating-depth": "Estimating depth is the first step.",
    "predicts-depth": "The model predicts depth from a single RGB image.",
    "computes-depth": "It computes per-pixel depth.",
    "infers-depth": "It infers dense depth from the photo.",
    "midas": "Powered by MiDaS weights.",
    "depth-anything": "Uses Depth Anything V2 under the hood.",
    "zoedepth": "ZoeDepth provides metric scale.",
}


@pytest.mark.parametrize(("case", "text"), list(CLAIM_DOCS.items()), ids=list(CLAIM_DOCS))
def test_documentation_claim_is_detected(case, text):
    assert claim_phrases(text), case
    assert denial_phrases(text) == []


NEUTRAL_DOCS = {
    "depth-first-search": "A depth-first search walks the scene graph.",
    "max-depth": "Set max depth to 8 to bound recursion.",
    "color-depth": "Images are converted to 8-bit color depth.",
    "nesting-depth": "The nesting depth of the config is limited.",
    "water-depth": "Logs the water depth sensor reading.",
    "in-depth": "See the in-depth guide for details.",
    "detection-only": "Runs object detection and instance segmentation.",
    "optical-flow-only": "Computes optical flow between frames.",
    "pose-only": "Estimates 2D human pose keypoints.",
    "classification-only": "Classifies images into 1000 categories.",
    "empty": "",
    "unrelated": "A command line tool for renaming files.",
}


@pytest.mark.parametrize(("case", "text"), list(NEUTRAL_DOCS.items()), ids=list(NEUTRAL_DOCS))
def test_neutral_documentation_makes_no_claim(case, text):
    assert claim_phrases(text) == [], case
    assert denial_phrases(text) == [], case


# --------------------------------------------------------------------------- #
# Documentation denials
# --------------------------------------------------------------------------- #

DENIAL_DOCS = {
    "no-depth-estimation": "There is no depth estimation in this package.",
    "no-monocular-depth-estimation": "No monocular depth estimation is performed.",
    "no-depth-maps": "The tool produces no depth maps.",
    "no-disparity-maps": "There are no disparity maps in the output.",
    "does-not-perform": "This tool does not perform depth estimation.",
    "does-not-support": "It does not support depth-estimation at all.",
    "does-not-implement": "The package does not implement any depth estimation.",
    "does-not-estimate": "It does not estimate depth.",
    "does-not-produce-depth-maps": "It does not produce depth maps.",
    "without": "Works without any depth estimation.",
    "not-a-depth-estimator": "This is not a depth estimator.",
    "never-estimates": "The auditor never estimates depth itself.",
    "is-not-supported": "Depth estimation is not supported here.",
    "cannot-estimate": "The tool cannot estimate depth.",
}


@pytest.mark.parametrize(("case", "text"), list(DENIAL_DOCS.items()), ids=list(DENIAL_DOCS))
def test_documentation_denial_is_detected(case, text):
    assert denial_phrases(text), case


@pytest.mark.parametrize(("case", "text"), list(DENIAL_DOCS.items()), ids=list(DENIAL_DOCS))
def test_denial_sentence_is_not_also_a_claim(case, text):
    assert claim_phrases(text) == [], case


# --------------------------------------------------------------------------- #
# Documentation structure
# --------------------------------------------------------------------------- #


def test_claim_line_number_is_reported():
    text = "# Title\n\nSome intro.\n\nIt estimates depth per frame.\n"
    claims, _ = scan_doc_text(text, "README.md")
    assert [hit.line for hit in claims] == [5]


def test_fenced_code_block_is_not_a_claim():
    text = 'Usage:\n\n```python\npipe = pipeline("depth-estimation")\n```\n\nNothing else.\n'
    assert claim_phrases(text) == []


def test_tilde_fenced_code_block_is_not_a_claim():
    text = "Usage:\n\n~~~\ndepth estimation example\n~~~\n"
    assert claim_phrases(text) == []


def test_text_after_fence_is_scanned_again():
    text = "```\ncode\n```\n\nIt estimates depth.\n"
    claims, _ = scan_doc_text(text, "README.md")
    assert [hit.line for hit in claims] == [5]


def test_wrapped_denial_sentence_is_detected():
    text = "This tool does not perform\ndepth estimation at all.\n"
    assert denial_phrases(text) == ["does-not-perform-depth-estimation"]
    assert claim_phrases(text) == []


def test_bullet_items_are_independent_units():
    text = "- Returns a depth map\n- Does not perform depth estimation\n"
    assert claim_phrases(text) == ["depth-map"]
    assert denial_phrases(text) == ["does-not-perform-depth-estimation"]


def test_one_claim_per_sentence():
    text = "It performs depth estimation and returns a depth map.\n"
    assert len(claim_phrases(text)) == 1


def test_doc_segments_skips_blank_lines():
    assert doc_segments("\n\n  \n") == []


def test_doc_segments_splits_sentences():
    segments = doc_segments("First one. Second one.\n")
    assert [text for text, _ in segments] == ["First one.", "Second one."]


def test_doc_segments_reports_heading_lines():
    segments = doc_segments("# Heading\n\nBody text.\n")
    assert segments == [("# Heading", 1), ("Body text.", 3)]


def test_claim_snippet_is_captured():
    claims, _ = scan_doc_text("It estimates depth per frame.\n", "README.md")
    assert claims[0].text == "It estimates depth per frame."
    assert claims[0].path == "README.md"


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #

GOOD_CODE = 'pipe = pipeline("depth-estimation")\ndepth = pipe(image)\n'
CLAIM_DOC = "This tool performs monocular depth estimation.\n"
DENIAL_DOC = "This tool does not perform depth estimation.\n"


def test_claim_backed_by_evidence_is_clean():
    result = audit_texts(docs=[("README.md", CLAIM_DOC)], code=[("pkg/a.py", GOOD_CODE)])
    assert result.findings == []
    assert result.status("high") == "pass"
    assert result.exit_code("low") == 0


def test_claim_without_evidence_is_high():
    result = audit_texts(docs=[("README.md", CLAIM_DOC)], code=[("pkg/a.py", "x = 1\n")])
    assert [finding.id for finding in result.findings] == [
        "documented-depth-estimation-claim-false"
    ]
    assert result.findings[0].severity == "high"
    assert result.exit_code("high") == 2


def test_claim_without_evidence_cites_the_claim():
    result = audit_texts(docs=[("README.md", CLAIM_DOC)], code=[("pkg/a.py", "x = 1\n")])
    (finding,) = result.findings
    assert finding.claims[0].path == "README.md"
    assert finding.evidence == ()


def test_claim_with_test_only_evidence_is_medium():
    result = audit_texts(
        docs=[("README.md", CLAIM_DOC)],
        code=[("pkg/a.py", "x = 1\n"), ("tests/test_a.py", GOOD_CODE)],
    )
    assert [finding.id for finding in result.findings] == [
        "documented-depth-estimation-claim-test-only"
    ]
    assert result.findings[0].severity == "medium"
    assert result.exit_code("high") == 0
    assert result.exit_code("medium") == 2


def test_denial_contradicted_by_evidence_is_high():
    result = audit_texts(docs=[("README.md", DENIAL_DOC)], code=[("pkg/a.py", GOOD_CODE)])
    assert [finding.id for finding in result.findings] == [
        "denied-depth-estimation-contradicted"
    ]
    assert result.findings[0].severity == "high"
    assert result.exit_code("high") == 2


def test_honest_denial_is_clean():
    result = audit_texts(docs=[("README.md", DENIAL_DOC)], code=[("pkg/a.py", "x = 1\n")])
    assert result.findings == []


def test_denial_with_test_only_evidence_is_clean():
    result = audit_texts(
        docs=[("README.md", DENIAL_DOC)], code=[("tests/test_a.py", GOOD_CODE)]
    )
    assert result.findings == []


def test_undocumented_evidence_is_low():
    result = audit_texts(docs=[("README.md", "A tool.\n")], code=[("pkg/a.py", GOOD_CODE)])
    assert [finding.id for finding in result.findings] == ["undocumented-depth-estimation"]
    assert result.findings[0].severity == "low"
    assert result.exit_code("high") == 0
    assert result.exit_code("low") == 2


def test_contradictory_documentation_is_medium():
    text = "- It estimates depth per frame\n- It does not perform depth estimation\n"
    result = audit_texts(docs=[("README.md", text)], code=[("pkg/a.py", GOOD_CODE)])
    ids = [finding.id for finding in result.findings]
    assert "contradictory-depth-estimation-documentation" in ids
    assert "denied-depth-estimation-contradicted" in ids


def test_findings_are_sorted_by_severity():
    text = "- It estimates depth per frame\n- It does not perform depth estimation\n"
    result = audit_texts(docs=[("README.md", text)], code=[("pkg/a.py", GOOD_CODE)])
    ranks = [SEVERITY_RANK[finding.severity] for finding in result.findings]
    assert ranks == sorted(ranks, reverse=True)


def test_no_docs_and_no_code_is_clean():
    result = audit_texts()
    assert result.findings == []
    assert result.summary()["claims"] == 0


def test_claim_with_near_miss_only_is_still_false():
    result = audit_texts(
        docs=[("README.md", CLAIM_DOC)],
        code=[("pkg/a.py", 'p = pipeline("depth-estimation")\np.eval()\n')],
    )
    assert [finding.id for finding in result.findings] == [
        "documented-depth-estimation-claim-false"
    ]


def test_claim_plus_denial_without_evidence_is_only_contradictory():
    text = "- It estimates depth per frame\n- It does not perform depth estimation\n"
    result = audit_texts(docs=[("README.md", text)], code=[("pkg/a.py", "x = 1\n")])
    assert [finding.id for finding in result.findings] == [
        "contradictory-depth-estimation-documentation"
    ]
    assert result.exit_code("high") == 0
    assert result.exit_code("medium") == 2


def test_claim_plus_denial_with_test_only_evidence_is_only_contradictory():
    text = "- It estimates depth per frame\n- It does not perform depth estimation\n"
    result = audit_texts(docs=[("README.md", text)], code=[("tests/test_a.py", GOOD_CODE)])
    assert [finding.id for finding in result.findings] == [
        "contradictory-depth-estimation-documentation"
    ]


def test_build_findings_accepts_raw_sequences():
    findings = build_findings([], [], [])
    assert findings == []


def test_max_severity_reports_highest():
    result = audit_texts(docs=[("README.md", CLAIM_DOC)], code=[("pkg/a.py", "x = 1\n")])
    assert result.max_severity() == "high"


def test_max_severity_is_none_when_clean():
    assert audit_texts().max_severity() is None


def test_real_and_test_evidence_split():
    result = audit_texts(
        code=[("pkg/a.py", GOOD_CODE), ("tests/test_a.py", GOOD_CODE)],
    )
    assert len(result.real_evidence) == 1
    assert len(result.test_evidence) == 1


# --------------------------------------------------------------------------- #
# Severity thresholds
# --------------------------------------------------------------------------- #


def _finding(severity: str) -> Finding:
    return Finding(
        id="undocumented-depth-estimation",
        severity=severity,
        title="t",
        detail="d",
    )


@pytest.mark.parametrize(
    ("severity", "fail_on", "expected"),
    [
        ("high", "high", 2),
        ("high", "medium", 2),
        ("high", "low", 2),
        ("high", "none", 0),
        ("medium", "high", 0),
        ("medium", "medium", 2),
        ("medium", "low", 2),
        ("medium", "none", 0),
        ("low", "high", 0),
        ("low", "medium", 0),
        ("low", "low", 2),
        ("low", "none", 0),
    ],
)
def test_exit_code_for_threshold(severity, fail_on, expected):
    assert exit_code_for([_finding(severity)], fail_on) == expected


def test_exit_code_for_no_findings():
    assert exit_code_for([], "low") == 0


def test_exit_code_for_rejects_unknown_threshold():
    with pytest.raises(AuditError):
        exit_code_for([], "critical")


# --------------------------------------------------------------------------- #
# Result serialisation
# --------------------------------------------------------------------------- #


def test_to_dict_shape():
    result = audit_texts(docs=[("README.md", CLAIM_DOC)], code=[("pkg/a.py", "x = 1\n")])
    data = result.to_dict("high")
    assert data["tool"] == "depth-estimation-claim-audit"
    assert data["version"] == "0.1.0"
    assert data["status"] == "fail"
    assert data["exit_code"] == 2
    assert data["max_severity"] == "high"
    assert data["summary"]["findings"] == 1
    assert data["findings"][0]["id"] == "documented-depth-estimation-claim-false"


def test_to_dict_is_json_serialisable():
    import json

    result = audit_texts(docs=[("README.md", CLAIM_DOC)], code=[("pkg/a.py", GOOD_CODE)])
    assert json.loads(json.dumps(result.to_dict("high")))["status"] == "pass"


def test_to_dict_respects_fail_on():
    result = audit_texts(code=[("pkg/a.py", GOOD_CODE)])
    assert result.to_dict("high")["status"] == "pass"
    assert result.to_dict("low")["status"] == "fail"


def test_summary_counts():
    result = audit_texts(
        docs=[("README.md", CLAIM_DOC)],
        code=[("pkg/a.py", GOOD_CODE), ("pkg/b.py", 'p = pipeline("depth-estimation")\n')],
    )
    summary = result.summary()
    assert summary == {
        "doc_files": 1,
        "code_files": 2,
        "claims": 1,
        "denials": 0,
        "evidence": 1,
        "near_misses": 1,
        "findings": 0,
    }


# --------------------------------------------------------------------------- #
# File collection and on-disk auditing
# --------------------------------------------------------------------------- #


@pytest.fixture()
def project(tmp_path):
    (tmp_path / "README.md").write_text(CLAIM_DOC, encoding="utf-8")
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    (pkg_dir / "depth.py").write_text(GOOD_CODE, encoding="utf-8")
    (pkg_dir / "notes.txt").write_text("Release notes live in the changelog.\n", encoding="utf-8")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "stale.py").write_text(GOOD_CODE, encoding="utf-8")
    return tmp_path


def test_collect_doc_files_finds_markdown_and_text(project):
    found = collect_doc_files([project])
    assert any(path.endswith("README.md") for path in found)
    assert any(path.endswith("pkg/notes.txt") for path in found)


def test_collect_code_files_finds_python(project):
    found = collect_code_files([project])
    assert [path for path in found if path.endswith("pkg/depth.py")]


def test_collect_skips_pycache(project):
    assert all("__pycache__" not in path for path in collect_code_files([project]))


def test_collect_respects_exclude(project):
    found = collect_code_files([project], exclude=["*/pkg/*"])
    assert found == []


def test_collect_excludes_by_basename(project):
    found = collect_doc_files([project], exclude=["README.md"])
    assert all(not path.endswith("README.md") for path in found)


def test_collect_missing_path_raises():
    with pytest.raises(AuditError):
        collect_code_files(["does/not/exist"])


def test_collect_single_file(project):
    found = collect_doc_files([project / "README.md"])
    assert len(found) == 1


def test_collect_force_accepts_any_suffix(tmp_path):
    odd = tmp_path / "manual.rest"
    odd.write_text(CLAIM_DOC, encoding="utf-8")
    assert collect_doc_files([odd]) == []
    assert len(collect_doc_files([odd], force=True)) == 1


def test_audit_paths_on_project(project):
    result = audit_paths([project])
    assert result.evidence
    assert result.claims
    assert result.findings == []


def test_audit_paths_flags_denial_in_a_second_doc(project):
    (project / "FAQ.md").write_text("No depth estimation is performed.\n", encoding="utf-8")
    result = audit_paths([project])
    ids = [finding.id for finding in result.findings]
    assert "denied-depth-estimation-contradicted" in ids
    assert "contradictory-depth-estimation-documentation" in ids


def test_audit_paths_classifies_explicit_docs_and_code(project):
    result = audit_paths(
        [], doc_paths=[project / "README.md"], code_paths=[project / "pkg" / "depth.py"]
    )
    assert len(result.doc_files) == 1
    assert len(result.code_files) == 1
    assert result.findings == []


def test_audit_paths_missing_path_raises():
    with pytest.raises(AuditError):
        audit_paths(["nope"])


def test_audit_paths_reports_false_claim(tmp_path):
    (tmp_path / "README.md").write_text(CLAIM_DOC, encoding="utf-8")
    (tmp_path / "main.py").write_text("print('hi')\n", encoding="utf-8")
    result = audit_paths([tmp_path])
    assert [finding.id for finding in result.findings] == [
        "documented-depth-estimation-claim-false"
    ]


def test_audit_paths_handles_undecodable_bytes(tmp_path):
    (tmp_path / "README.md").write_bytes(b"It estimates depth \xff\xfe now.\n")
    (tmp_path / "main.py").write_bytes(b"x = 1  # \xff\n")
    result = audit_paths([tmp_path])
    assert result.claims
