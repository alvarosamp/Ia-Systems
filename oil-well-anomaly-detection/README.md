# Oil Well Anomaly Detection

Anomaly detection on offshore oil wells using the [Petrobras 3W dataset](https://github.com/petrobras/3W).
Focus problem: **Spurious Closure of DHSV** (Downhole Safety Valve, event type **2**).

This repo lives in the [`Ia-Systems`](https://github.com/alvarosamp/Ia-Systems) monorepo and reuses ~80% of the
template established in `turbofan-rul-prediction` (settings, schemas, Hydra layout, Makefile, CI).

> **Status:** Phase 1 — project skeleton, data acquisition, validated preparation pipeline.
> Modeling (XGBoost + LSTM-Autoencoder), training, deployment, monitoring and explainability arrive in later phases.

---

## Problem

Each 3W instance is a multivariate time series sampled at 1 Hz from eight sensors:

| Channel       | Meaning                                     |
| ------------- | ------------------------------------------- |
| `P-PDG`       | Permanent Downhole Gauge pressure           |
| `P-TPT`       | Temperature/Pressure Transducer pressure    |
| `T-TPT`       | Temperature/Pressure Transducer temperature |
| `P-MON-CKP`   | Monitoring choke pressure                   |
| `T-JUS-CKP`   | Downstream choke temperature                |
| `P-JUS-CKGL`  | Downstream gas-lift choke pressure          |
| `T-JUS-CKGL`  | Downstream gas-lift choke temperature       |
| `QGL`         | Gas-lift flow rate                          |

Labels (`class` column, `Int64`):

| Code  | Meaning                                  |
| ----- | ---------------------------------------- |
| `0`   | Normal steady-state                      |
| `2`   | DHSV closure (anomaly, steady-state)     |
| `102` | DHSV closure transient (run-up to fault) |

Phase 1 keeps these three labels and discards everything else.

---

## Quickstart

```bash
# 1. Install
make install-dev

# 2. Download the relevant slice of 3W (event dirs 0 + 2 only — fast, ~hundreds of MB)
make download-data

# 3. Build data/processed/
make prepare-data

# 4. Run the test suite
make test
```

Smoke test the pipeline on three files only:

```bash
make prepare-data-smoke
```

---

## Layout

```
oil-well-anomaly-detection/
├── conf/                 # Hydra config groups
│   ├── config.yaml
│   ├── paths/default.yaml
│   ├── data/3w.yaml
│   └── features/default.yaml
├── data/                 # gitignored — raw/, interim/, processed/
├── src/oil_well_anomaly/
│   ├── settings.py       # pydantic-settings (paths, sensor list, label codes)
│   ├── schemas.py        # pandera schemas (raw + processed)
│   └── data/
│       ├── download.py   # sparse-checkout of github.com/petrobras/3W
│       └── prepare.py    # filter → tag → validate → write parquets
├── scripts/              # Hydra entrypoints
│   ├── download_3w.py
│   └── prepare_data.py
├── tests/                # pandera + pipeline tests
└── Makefile
```

### Two CLIs, one job

There are two ways to invoke each step:

* **Thin CLI** (Typer) — `ow-download`, `ow-prepare`. Fast, no Hydra overhead, ideal for ad-hoc runs.
* **Hydra entrypoint** (`scripts/*.py`) — composable configs, multirun, hydra outputs/. Use when sweeping or wiring into experiments.

Both call the exact same library functions (`download_3w` / `prepare_dataset`).

---

## Design notes

* **Sparse git checkout** — the 3W repo carries the toolkit + dataset together, but we only want the data. `download.py` issues `git clone --filter=blob:none --sparse` and `git sparse-checkout set` so we pull only the requested event dirs at a pinned ref. Re-runs do an idempotent fetch + reset.
* **Per-instance parquet output** — each 3W instance is an independent time series. Keeping them separate (not concatenated) preserves the natural unit for cross-validation, lets us stream them lazily, and mirrors the upstream layout. A `_summary.csv` in `data/processed/` indexes them.
* **Two-stage schema** — `raw_schema` is permissive (nullable sensors, any 3W label) and runs right after `pd.read_parquet`. `processed_schema` is strict (label ∈ {0, 2, 102}, `instance_id` + `source_event_dir` required) and runs at the end of `prepare_instance`. If the upstream format ever drifts, the failure surface is small and obvious.
* **Single source of truth for constants** — sensor names, label codes and target labels live in `settings.py`. The Hydra config (`conf/data/3w.yaml`) mirrors these for runtime overrides. Bumping one without the other is caught by tests.
* **Failure isolation** — `prepare_instance` never raises on data issues. It records the reason in the report and the batch keeps going, so a single bad file can't take down a 1k-instance run.

---

## Configuration

Three layers, in order of precedence:

1. **CLI overrides** (Hydra) — `python scripts/prepare_data.py data.event_dirs=[0,2,5]`
2. **Environment variables / `.env`** — `OIL_WELL__PATHS__DATA_ROOT=/mnt/...`
3. **Defaults in `settings.py` and `conf/`**

See [`.env.example`](.env.example) for the full list of supported env vars.

---

## Roadmap

| Phase | Scope                                                         |
| ----- | ------------------------------------------------------------- |
| **1** | Skeleton, download, prepare, schemas, tests **← you are here** |
| 2     | Feature engineering (windows, rolling stats), train/val/test split, baseline metrics |
| 3     | Supervised model (XGBoost) + unsupervised (LSTM-Autoencoder), comparison framework |
| 4     | MLOps wiring (MLflow tracking, model registry, DVC for data)  |
| 5     | API + CI/CD (FastAPI, GitHub Actions, Render deploy)          |
| 6     | Monitoring + explainability (Evidently, SHAP)                 |
