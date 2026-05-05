"""Prepare raw 3w parquet files for downstream pipeline
    
    Pipeline
    
    ----
    For eah instance file under the requested event-type folders:
    1. Load the parquet(timestamp index, float sensors, 'int64', 'class')
    2. Validate argains : data 'core.schemas.raw_schema'
    3. Drop rows whose 'class' is missing or outside :data'target_labels'
    4. Sort the index, drop duplicates timestamps (keeping first), and tag
    the dataframe with instace_id + source_event_dir
    5. validate against: data 'processed_schema'
    6. Write to ``data/processed/{event_dir}/{instance_id}.parquet`` using
   the same engine/compression as the upstream dataset (``pyarrow`` +
   ``brotli``) so files are interchangeable.
 
A run summary CSV is written to ``data/processed/_summary.csv`` with one
row per processed instance: row count per label, time span, source.
 
Why per-instance files (not one big parquet)
--------------------------------------------
Each 3W instance is an independent multivariate time series. Keeping
them separate (a) preserves the natural unit for cross-validation
splits, (b) lets us stream them lazily during training, and (c) mirrors
the upstream layout, which makes diffing and re-runs cheap.
 
CLI
---
::
 
    python -m data.prepare
    python -m data.prepare data.prepare.max_instances=5
    python -m data.prepare data.event_dirs=[0,2] data.prepare.validate=false
 
Or via the installed shortcut::
 
    ow-prepare
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import logging
import hydra
import pandas as pd
import pandera as pa
from omegaconf import DictConfig
from tqdm import tqdm

from core.schemas import processed_schema, raw_schema
from core.settings import EVENT_DIRS_FOR_DHSV, LABEL_COL, settings, TARGET_LABELS, SENSORS

logger = logging.getLogger(__name__)

#Result types

@dataclass(slots = True)
class InstanceReport:
    '''per instance outcome'''
    instance_id: str
    source_event_dir: str
    src_path: Path
    n_rows_in: int
    out_path: Path | None = None
    n_rows_out: int | None = None
    label_counts: dict[int, int] = field(default_factory=dict)
    t_start: pd.Timestamp | None = None
    t_end: pd.Timestamp | None = None
    skipped: bool = False
    skip_reason: str = ""
    
@dataclass(slots = True)
class PrepareSummary:
    """Agreggate result of a prepare run"""
    reports : list[InstanceReport] = field(default_factory = list)
    @property
    def n_instances(self) -> int:
        return len(self.reports)
    
    @property
    def n_kept(self) -> int:
        return sum(1 for r in self.reports if not r.skipped)
    
    @property
    def n_skipped(self) -> int:
        return sum(1 for r in self.reports if r.skipped)
    
    def to_frame(self) -> pd.DataFrame:
        rows = []
        for r in self.reports:
            row = {
                "instance_id": r.instance_id,
                "source_event_dir": r.source_event_dir,
                "src_path": str(r.src_path),
                "out_path": str(r.out_path) if r.out_path else "",
                "n_rows_in": r.n_rows_in,
                "n_rows_out": r.n_rows_out,
                "t_start": r.t_start,
                "t_end": r.t_end,
                "skipped": r.skipped,
                "skip_reason": r.skip_reason,
            }
            for label in TARGET_LABELS:
                row[f"n_label_{label}"] = r.label_counts.get(label, 0)
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Single-file pipeline
# ---------------------------------------------------------------------------


def _read_raw(path: Path, *, validate: bool) -> pd.DataFrame:
    """Read a 3W parquet and optionally validate against the raw schema."""
    df = pd.read_parquet(path, engine="pyarrow")

    # Some 3W files have missing values in the label column; we drop them later.
    keep = [*SENSORS, LABEL_COL]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    df = df[keep].copy()

    # The 3W spec mandates Int64 for class. Parquet round-trips can downgrade
    # it to plain int64 — coerce defensively.
    if not pd.api.types.is_extension_array_dtype(df[LABEL_COL]):
        df[LABEL_COL] = df[LABEL_COL].astype("Int64")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="raise")

    if validate:
        df = raw_schema.validate(df, lazy=True)
    return df

def filter_and_tag(
    df: pd.DataFrame,
    *,
    instance_id: str,
    source_event_dir: str, 
) ->  pd.DataFrame:
    """Drop rows outside the DHSV label set and tag instance metadata."""
    mask = df[LABEL_COL].isin(list(TARGET_LABELS)) & df[LABEL_COL].notna()
    out = df.loc[mask].copy()
 
    # Sort + dedup. Some 3W instances have repeated timestamps from the
    # original SCADA exports; keep the first occurrence.
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out.index.name = "timestamp"
 
    out["instance_id"] = instance_id
    out["source_event_dir"] = source_event_dir
    return out

def prepare_instance(
    src_path: Path,
    *,
    event_dir: str,
    out_root: Path,
    validate: bool = True,
) -> InstanceReport:
    """Process a single 3W instance file. Returns a report (never raises
    on data issues — failures are recorded in ``skip_reason``)."""
    instance_id = src_path.stem
    n_in = 0
 
    try:
        raw = _read_raw(src_path, validate=validate)
        n_in = len(raw)
        prepared = filter_and_tag(raw, instance_id=instance_id, source_event_dir=event_dir)
 
        if prepared.empty:
            return InstanceReport(
                instance_id=instance_id,
                source_event_dir=event_dir,
                src_path=src_path,
                out_path=None,
                n_rows_in=n_in,
                n_rows_out=0,
                label_counts={},
                t_start=None,
                t_end=None,
                skipped=True,
                skip_reason="no rows in target label set",
            )
 
        if validate:
            prepared = processed_schema.validate(prepared, lazy=True)
 
        out_dir = out_root / event_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{instance_id}.parquet"
        prepared.to_parquet(
            out_path,
            engine="pyarrow",
            compression="brotli",
            index=True,
        )
 
        counts = (
            prepared[LABEL_COL]
            .value_counts(dropna=False)
            .astype(int)
            .to_dict()
        )
        return InstanceReport(
            instance_id=instance_id,
            source_event_dir=event_dir,
            src_path=src_path,
            out_path=out_path,
            n_rows_in=n_in,
            n_rows_out=len(prepared),
            label_counts={int(k): int(v) for k, v in counts.items()},
            t_start=prepared.index.min(),
            t_end=prepared.index.max(),
        )
 
    except (pa.errors.SchemaErrors, pa.errors.SchemaError) as exc:
        logger.warning("Schema check failed for %s: %s", src_path.name, exc)
        return InstanceReport(
            instance_id=instance_id,
            source_event_dir=event_dir,
            src_path=src_path,
            out_path=None,
            n_rows_in=n_in,
            n_rows_out=0,
            label_counts={},
            t_start=None,
            t_end=None,
            skipped=True,
            skip_reason=f"schema_error: {type(exc).__name__}",
        )
    except Exception as exc:  # noqa: BLE001 — per-file safety net
        logger.exception("Failed to prepare %s", src_path)
        return InstanceReport(
            instance_id=instance_id,
            source_event_dir=event_dir,
            src_path=src_path,
            out_path=None,
            n_rows_in=n_in,
            n_rows_out=0,
            label_counts={},
            t_start=None,
            t_end=None,
            skipped=True,
            skip_reason=f"error: {exc}",
        )
 
 
# ---------------------------------------------------------------------------
# Batch pipeline
# ---------------------------------------------------------------------------
 
 
def _iter_instances(
    dataset_root: Path,
    event_dirs: Iterable[str],
) -> list[tuple[str, Path]]:
    """List ``(event_dir, parquet_path)`` pairs for the requested folders."""
    pairs: list[tuple[str, Path]] = []
    for ed in event_dirs:
        ed_path = dataset_root / ed
        if not ed_path.exists():
            logger.warning("Event dir not found, skipping: %s", ed_path)
            continue
        for f in sorted(ed_path.glob("*.parquet")):
            pairs.append((ed, f))
    return pairs
 
 
def prepare_dataset(
    event_dirs: Iterable[str] = EVENT_DIRS_FOR_DHSV,
    *,
    dataset_root: Path | None = None,
    out_root: Path | None = None,
    max_instances: int | None = None,
    validate: bool = True,
) -> PrepareSummary:
    """Run the full prepare pipeline and write a summary CSV."""
    settings.paths.ensure()
    dataset_root = dataset_root or settings.dataset_path
    out_root = out_root or settings.paths.processed_dir
 
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"3W dataset not found at {dataset_root}. "
            "Did you run `make download-data`?"
        )
 
    pairs = _iter_instances(dataset_root, event_dirs)
    if max_instances is not None:
        pairs = pairs[:max_instances]
 
    if not pairs:
        logger.warning(
            "No parquet files found under %s for event dirs %s",
            dataset_root,
            list(event_dirs),
        )
        return PrepareSummary()
 
    logger.info(
        "Preparing %s instances from %s into %s",
        len(pairs),
        dataset_root,
        out_root,
    )
 
    summary = PrepareSummary()
    for event_dir, src_path in tqdm(pairs, desc="prepare", unit="file"):
        report = prepare_instance(
            src_path,
            event_dir=event_dir,
            out_root=out_root,
            validate=validate,
        )
        summary.reports.append(report)
 
    summary_path = out_root / "_summary.csv"
    summary.to_frame().to_csv(summary_path, index=False)
    logger.info(
        "Done. %s prepared, %s skipped. Summary -> %s",
        summary.n_kept,
        summary.n_skipped,
        summary_path,
    )
    return summary

#hydra entry point 
@hydra.main(version_base = None, config_path = '../../configs', config_name = 'config')
def main(cfg : DictConfig) -> None:
    """ hydra entry point. See module docstring for usage"""
    logger.info("Config:\n%s", cfg)

    data_cfg = cfg.get("data", {})
    paths_cfg = cfg.get("paths", {})
    prepare_cfg = data_cfg.get("prepare", {}) if hasattr(data_cfg, "get") else {}

    event_dirs = list(data_cfg.get("event_dirs", EVENT_DIRS_FOR_DHSV)) if hasattr(data_cfg, "get") else list(EVENT_DIRS_FOR_DHSV)
    dataset_root = Path(paths_cfg.get("dataset_dir", settings.dataset_path)) if hasattr(paths_cfg, "get") else settings.dataset_path
    out_root = Path(paths_cfg.get("processed_dir", settings.paths.processed_dir)) if hasattr(paths_cfg, "get") else settings.paths.processed_dir
    max_instances = prepare_cfg.get("max_instances", None) if hasattr(prepare_cfg, "get") else None
    validate_flag = bool(prepare_cfg.get("validate", True)) if hasattr(prepare_cfg, "get") else True

    summary = prepare_dataset(
        event_dirs=event_dirs,
        dataset_root=dataset_root,
        out_root=out_root,
        max_instances=max_instances,
        validate=validate_flag,
    )
    logger.info(
        "Summary: %s prepared, %s skipped (out of %s)",
        summary.n_kept,
        summary.n_skipped,
        summary.n_instances,
    )
 
 
if __name__ == "__main__":
    main()
