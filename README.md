# BEAR: Benefit-Aware Expert Routing for Financial Table Recognition

BEAR predicts
the replacement benefit of calling a stronger table-recognition expert from
observations available after the Primary recognizer runs. It ranks a batch of
tables and spends an exact expert-call budget on the highest predicted
benefits.

Primary runs on every table. A successfully parsed expert result replaces the
complete Primary table, even when the replacement is harmful. A generation or
parsing failure retains Primary but still consumes the call. Routing is
batch-level and uses exactly the requested number of calls. Call counts are
expert calls, not a measure of hardware-normalized compute.

## Installation

Use Python 3.12.

On Ubuntu 24.04, install the native build prerequisites required to build
`pylcs`, a GriTS dependency:

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3.12-dev
```

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt
```

Expert inference requires CUDA; HunyuanOCR-1.5 additionally requires
bfloat16 support. HunyuanOCR-1.5 and GLM-OCR weights are downloaded at
runtime from their model repositories. Model weights remain subject to their
upstream licenses. The VietFinTab development router stage
produces the HGB router required by final VietFinTab routing and the TiniX
study. Run that stage before TiniX routing.

## Reproduce VietFinTab

Create the cohort summary:

```bash
python reproduce.py --study vietfintab --stage split
```

Run Primary on the calibration, development, and final cohorts:

```bash
python reproduce.py --study vietfintab --stage primary --cohort calibration
python reproduce.py --study vietfintab --stage primary --cohort development
python reproduce.py --study vietfintab --stage primary --cohort final
```

Run Hunyuan on the development cohort. This stage requires CUDA:

```bash
python reproduce.py --study vietfintab --stage expert \
  --expert hunyuanocr_1_5 --cohort development --device cuda
```

Evaluate the development outputs:

```bash
python reproduce.py --study vietfintab --stage evaluate \
  --system primary --cohort development
python reproduce.py --study vietfintab --stage evaluate \
  --system hunyuanocr_1_5 --cohort development
```

Fit the development routers, then create the final route before producing
any final expert or evaluation output:

```bash
python reproduce.py --study vietfintab --stage router --split development
python reproduce.py --study vietfintab --stage router --split final
```

Run the final experts. These stages require CUDA:

```bash
python reproduce.py --study vietfintab --stage expert \
  --expert hunyuanocr_1_5 --cohort final --device cuda
python reproduce.py --study vietfintab --stage expert \
  --expert glm_ocr --cohort final --device cuda
```

Evaluate the final outputs:

```bash
python reproduce.py --study vietfintab --stage evaluate \
  --system primary --cohort final
python reproduce.py --study vietfintab --stage evaluate \
  --system hunyuanocr_1_5 --cohort final
python reproduce.py --study vietfintab --stage evaluate \
  --system glm_ocr --cohort final
```

Assemble the final report:

```bash
python reproduce.py --study vietfintab --stage report --split final
```

Router settings are selected using development data. The selected predictor
is then refit on all 600 development tables. Final tables are ranked using
only Primary-side observations, and final labels are not used for model
selection. The unchanged Hunyuan-trained BEAR ranking is reused for the
GLM transfer comparison; no GLM-specific router is fit or calibrated.

The final split holds out tables from the same 13 issuers as development. It
is table-held-out, not issuer-held-out. Recognition and evaluation records
can be resumed from their generated files. Use `--data-dir`, `--models-dir`,
and `--output-dir` to choose storage locations.

## External evaluation on TiniX

The exact 52-report cohort is listed in `manifests/tinix.json`. The workflow
downloads the reports and OCR sources from the pinned TiniX dataset, processes
2,319 pages, detects tables on every page, and applies the unchanged
development-trained HGB ranking to all 2,275 detected tables.

Run the stages in order:

```bash
python reproduce.py --study tinix --stage prepare
python reproduce.py --study tinix --stage primary
python reproduce.py --study tinix --stage route
python reproduce.py --study tinix --stage expert --device cuda
python reproduce.py --study tinix --stage evaluate
python reproduce.py --study tinix --stage report
```

`prepare` verifies 52 reports, 50 reporting entities, 2,319 pages, and the
explicit OCR page markers. `primary` detects tables and extracts the 13
Primary-side routing features. `route` selects 455 expert calls. `expert`
runs HunyuanOCR-1.5 and retains Primary after a failed generation or parse.
`evaluate` creates the deterministic OCR-silver alignment and computes
GriTS-Con, GriTS-Top, and report-clustered paired intervals. The stages
resume from ordinary files under `outputs/tinix/`. Resume existing TiniX
outputs only when models, settings, and upstream outputs are unchanged; use a
fresh output directory when those inputs change.

Evaluation uses 832 aligned tables from 51 reports: 149 are routed and 683
are not routed. TiniX references are OCR-generated silver references, not
manually verified human gold. Alignment is restricted to pages with one
explicit page marker, one top-level OCR HTML table, one detected table, and a
canonicalizable OCR table. This restriction may introduce selection bias. The
scores do not establish accounting correctness.

## Reproduce Figure 2

After the VietFinTab final report exists, render Figure 2 from the calculated
results:

```bash
python reproduce.py --study vietfintab --stage figure
```

The output is `outputs/vietfintab/results/figure2.svg`. The Oracle curve uses
the exact-B whole-table replacement semantics used by the experiment.

## Additional analyses

### Routing diagnostics

Run the VietFinTab development and final reproduction stages first. This
analysis reuses their router, recognition, and evaluation outputs. It does
not rerun Primary, HunyuanOCR-1.5, or GLM-OCR.

```bash
python analysis/routing_diagnostics.py
```

The result is written to
`outputs/analysis/routing_diagnostics.json`. It covers feature-set controls,
paired feature-set contrasts, fitted-router permutation reliance,
BEAR vs. Difficulty selection agreement, and expert outcome characterization.

### Local-repair oracle

```bash
python analysis/local_repair_oracle.py
```

This analysis performs no model inference or GriTS rescoring. It recomputes
whole-table-only and local-enabled reference-assisted oracle frontiers from
per-action scores. The result is written to
`outputs/analysis/local_repair_oracle.json`.

The local-repair analysis starts from the recorded per-action scores in
`analysis/local_repair_actions.json` and does not rerun local expert
inference. It measures the tested bounded action family on the development
population and is not a deployable policy or a general claim about local
repair.

## Expected results

The following rounded values are reference checks for the tested environment.
Runtime outputs are calculated from the executed pipeline.

### VietFinTab

| System | GriTS-Con | GriTS-Top |
| --- | ---: | ---: |
| Primary | 0.8581 | 0.9256 |
| BEAR (HGB) + Hunyuan, 60 calls | 0.9092 | 0.9582 |
| unchanged BEAR ranking + GLM, 60 calls | 0.9100 | 0.9632 |

Hunyuan standalone coverage is 0.87. Expert failure retains Primary. At 60
calls, BEAR (HGB) versus matched random differs by about `+0.0458` GriTS-Con
with a 95% interval of `[0.0323, 0.0601]`. The direct HGB-versus-Ridge and
benefit-versus-Difficulty comparisons are not treated as decisive claims.
Difficulty (HGB) is a retrospective control using the same Primary features
and fixed HGB settings to predict Primary error rather than replacement
benefit.

### TiniX

| System | GriTS-Con | GriTS-Top |
| --- | ---: | ---: |
| Primary | 0.6710 | 0.8427 |
| BEAR (HGB) + Hunyuan | 0.7084 | 0.8620 |

The study contains 52 reports from 50 reporting entities, 2,319 pages, 2,275
detected tables, 455 expert calls, 832 aligned/scored tables, and 51 aligned
reports. Among the scored tables, 149 are routed.

| Paired difference | Estimate | 95% interval |
| --- | ---: | ---: |
| GriTS-Con | +0.0374 | [0.0211, 0.0562] |
| GriTS-Top | +0.0193 | [0.0082, 0.0319] |

## Data and models

Exact dataset and model revisions are pinned in `protocol.json` and the
manifests.

- VietFinTab supplies the development and final experiments.
- TiniX supplies the external report and OCR data.
- Primary uses RapidOCR, Docling, TableFormer, and Layout Heron.
- HunyuanOCR-1.5 provides expert outputs used for router development and evaluation.
- GLM-OCR supplies the transfer comparison.

Datasets, model weights, and financial-report PDFs remain subject to their
own terms. BEAR does not redistribute third-party weights or report files.

## Reproducibility notes

The tested environment reproduces the reported rounded metrics. Minor runtime
or floating-point differences may occur across hardware and software
environments. The workflow preserves the deterministic seed values used for
the reported random routes and bootstrap intervals.

## Citation

Khai Nguyen Bui and Tuan-Dung Cao, “BEAR: Benefit-Aware Expert Routing for
Financial Table Recognition.”

## License

BEAR source code is released under the [MIT License](LICENSE). Dataset and
model assets are subject to their respective upstream licenses.
