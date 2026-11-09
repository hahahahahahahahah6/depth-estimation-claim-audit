# depth-estimation-claim-audit

Audit what a project's documentation *says* about image depth estimation against
what its Python sources *do*.

A README that advertises "monocular depth estimation" while the code only loads a
model and never calls it, or only runs it from a test, is a claim a reviewer has
to check by hand. This tool checks it mechanically, offline, in milliseconds.

It is a static text auditor. It does not perform depth estimation, it never
imports or executes the audited project, it downloads no weights, and it makes no
network calls.

```console
$ depth-estimation-claim-audit .
depth-estimation-claim-audit 0.1.0
scanned 1 doc file(s), 3 python file(s)
claims: 2  denials: 0  evidence: 0  near-misses: 1
  near-miss scene/depth.py:12 [hf-pipeline-depth-estimation] pipeline("depth-estimation") (loader-without-invocation)

FAIL: 1 finding(s) at or above --fail-on high

  [HIGH] documented-depth-estimation-claim-false
    Documented depth-estimation claim is not backed by code
    2 documentation sentence(s) advertise depth estimation, but no
    depth-estimation evidence was found in 3 scanned Python file(s).
      claim: README.md:3 [depth-estimation] Scene Tool performs monocular depth estimation on every frame.
$ echo $?
2
```

## Install

```console
$ pip install .
```

Python 3.9+, no runtime dependencies.

## Usage

```console
$ depth-estimation-claim-audit [PATH ...] [--docs PATH] [--code PATH]
                              [--exclude GLOB] [--fail-on LEVEL] [--json] [--quiet]
$ python3 -m depth_estimation_claim_audit .
```

| Option | Meaning |
| --- | --- |
| `PATH ...` | Files or directories to audit, classified by suffix (default `.`). |
| `--docs PATH` | Force `PATH` to be read as documentation. Repeatable. |
| `--code PATH` | Force `PATH` to be read as Python source. Repeatable. |
| `--exclude GLOB` | Skip matching paths. Repeatable. |
| `--fail-on LEVEL` | `none`, `low`, `medium` or `high` (default `high`). |
| `--json` | Emit the full result as JSON. |
| `--quiet` | Print findings only, without the scan summary. |
| `--version` | Print `0.1.0`. |

Documentation suffixes: `.md`, `.markdown`, `.mdown`, `.rst`, `.txt`, `.adoc`.
Code suffixes: `.py`, `.pyi`. Virtualenvs, `__pycache__`, `build/`, `dist/`,
`node_modules/` and similar directories are skipped.

### Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Nothing at or above the `--fail-on` threshold. |
| `2` | At least one finding at or above the threshold. |
| `1` | Bad usage, or a path that does not exist. |

## What counts as evidence

A depth estimator must be **constructed and then invoked**. Loading a checkpoint
is not evidence; training it is not evidence.

| Backend | Construction | Accepted invocation |
| --- | --- | --- |
| transformers pipeline | `pipeline("depth-estimation")`, positional or `task=` | `pipe(...)`, `pipe.predict(...)` |
| transformers model | any `*ForDepthEstimation` class (`AutoModelForDepthEstimation`, `DPTForDepthEstimation`, `GLPNForDepthEstimation`, `ZoeDepthForDepthEstimation`, ...) | `model(...)`, `.forward`, `.predict`, `.infer`, ... |
| Depth Anything | `DepthAnything`, `DepthAnythingV2`, `DepthAnything*` | as above, including `.infer_image` |
| MiDaS | `torch.hub.load("intel-isl/MiDaS", ...)` or `"isl-org/MiDaS"`; `MidasNet`, `MidasNet_small`, `DPTDepthModel` | as above |
| ZoeDepth | `torch.hub.load("isl-org/ZoeDepth", ...)`; `ZoeDepth*`, `ZoeD_N`, `ZoeD_NK` | as above, including `.infer_pil` |
| OpenCV stereo | `cv2.StereoBM_create`, `cv2.StereoSGBM_create`, `cv2.StereoBM.create`, `cv2.StereoSGBM.create` | `.compute(...)` **only** |

Chained forms are understood: `pipeline("depth-estimation")(image)` and
`cv2.StereoBM_create(16, 15).compute(left, right)`.

In addition, calling any of 24 unambiguously depth-named functions is evidence on
its own — `estimate_depth`, `predict_depth`, `compute_depth`, `infer_depth`,
`depth_map`, `disparity_map`, `monocular_depth`, `generate_depth_map`, and so on.
Defining one of them is not: `def estimate_depth(...)` without a call is nothing.

### What is deliberately rejected

* Loading without invoking (`pipeline("depth-estimation")` and nothing else).
* Training only (`model.train()`, `load_state_dict`, `save_pretrained`, `.to(device)`, `.eval()`).
* The wrong pipeline task — `image-classification`, `object-detection`,
  `image-segmentation`, `optical-flow`, `pose-estimation` and friends.
* Non-literal model paths: `torch.hub.load(repo, "DPT_Large")` where `repo` is a
  variable or an f-string. Without a literal repository id there is nothing to
  verify, so nothing is claimed.
* Neighbouring vision tasks on their own: detection, segmentation,
  classification, optical flow, pose estimation.
* Anything inside a comment, a docstring or a string constant.
* Words where "depth" means something else: `max_depth`, `depth_first_search`,
  `color_depth`, `bit_depth`, `recursion_depth`.

Constructed-but-never-invoked estimators are still shown, as **near misses**.
They are context for a human, never evidence, and never a finding.

## Findings

| Id | Severity | Fires when |
| --- | --- | --- |
| `documented-depth-estimation-claim-false` | high | Docs advertise depth estimation, no evidence anywhere, and the docs never deny it. |
| `denied-depth-estimation-contradicted` | high | Docs state depth estimation is not performed, but shipped code performs it. |
| `documented-depth-estimation-claim-test-only` | medium | Docs advertise depth estimation; every evidence site is inside test files. |
| `contradictory-depth-estimation-documentation` | medium | The same documentation set both advertises and denies depth estimation. |
| `undocumented-depth-estimation` | low | Shipped code performs depth estimation; no scanned documentation mentions it. |

A document that explicitly denies depth estimation is never reported as making a
false capability claim — the contradiction is reported instead, at medium. No
reader of such a document is misled into expecting depth maps.

## Differentiation

* **Not a linter.** Linters judge code against code. This judges prose against
  code, which is where capability claims actually live.
* **Not a test runner.** It needs no GPU, no weights, no fixtures, and no ability
  to import the project. It works on a checkout that cannot even be installed.
* **Not a model evaluator.** It says nothing about whether the depth maps are
  *good*. It only answers whether the operation happens at all.
* **Evidence requires invocation.** Tools that grep for `import torch` or for the
  string `depth` cannot tell a loaded-and-unused model from a working one. The
  construct-then-invoke rule is the whole point.
* **Test-only evidence is called out separately.** "It works, in the test suite"
  is a different failure from "it does not exist", and gets its own severity.
* **Denials are first-class.** Documentation that promises *not* to do something
  is auditable too, and the audit fails when shipped code breaks that promise.
* **Offline and deterministic.** No network, no imports, no model downloads; the
  same input always yields the same JSON.

## Honest limitations

This tool matches exact phrases and literal code shapes. It does not understand
meaning. Specifically:

* **Phrase matching is not semantics.** Claims are found by a fixed phrase table
  (`depth estimation`, `depth map`, `predicts depth`, `MiDaS`, ...). A README
  that promises depth estimation in words outside that table — in another
  language, or as "recovers 3D structure per pixel" — is missed entirely.
* **Word-sense collisions cut both ways.** "Depth" is an overloaded English word.
  Tree depth, recursion depth, colour depth and water depth are excluded by name,
  but a document about a *different* kind of depth may still trip a claim phrase,
  and a document *about* depth estimation tooling — including this README — reads
  as a claim even when nothing performs depth estimation. That is why an explicit
  denial downgrades the finding instead of being overruled by it.
* **No data flow or scope analysis.** A model loaded in one function and called
  in another counts, because the whole file is searched for the call. A model
  returned from a factory and called through a different name does not count. An
  invocation guarded by `if False:` or sitting in dead code still counts.
* **Only Python, only text.** Depth estimation performed in C++, CUDA, ONNX
  Runtime, a notebook, a subprocess, a shell script, or a remote API is invisible.
  `.ipynb` files are not scanned.
* **Fenced code blocks in documentation are ignored.** A README whose only
  mention of depth estimation is a usage example makes no claim as far as this
  tool is concerned. That is a deliberate bias toward silence.
* **Sentence splitting is naive.** Denials and claims are matched per sentence,
  split on `.`, `!`, `?`, `;` and `:`, with list items and headings kept
  separate. A denial spread across two bullets, or phrased as a table cell, may
  not shield the claim next to it.
* **Test detection is path-based.** A file counts as a test when its path
  contains `test`/`tests`/`testing` or its name looks like `test_*.py` /
  `*_test.py` / `conftest.py`. Projects that keep tests elsewhere will have that
  evidence counted as shipped code.
* **The pattern tables lag the ecosystem.** New model families and new hub
  repository names are not recognised until they are added. When in doubt the
  tool stays quiet, so a false negative is far more likely than a false alarm.

## Development

```console
$ python3 -m pytest
```

363 unit tests and 38 CLI tests, no network, no fixtures beyond `tmp_path`.
A run that finds only sub-threshold findings prints them and still exits `0`.

## License

MIT. Copyright (c) 2026 Hao Li.
