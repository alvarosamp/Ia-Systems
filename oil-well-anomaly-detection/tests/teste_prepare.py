"""Prepare-pipeline tests.

We write a synthetic parquet that mimics the upstream 3W layout, run the
pipeline against a temp directory, and assert on the output.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.settings import LABEL_COL, SENSORS
from data.prepare import prepare_dataset, prepare_instance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_fake_3w_file(
    path: Path,
    *,
    labels: list[int | None],
) -> None:
    """Write a parquet that looks like a 3W instance."""
    n = len(labels)
    idx = pd.date_range("2024-06-01", periods=n, freq="1s")
    df = pd.DataFrame(
        {s: [float(i) for i in range(n)] for s in SENSORS},
        index=idx,
    )
    df[LABEL_COL] = pd.array(labels, dtype="Int64")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="brotli", index=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_dataset(tmp_path: Path) -> Path:
    """Build a tiny fake ``dataset/`` tree with two event dirs."""
    ds = tmp_path / "dataset"

    # Event dir "0" — pure normal.
    _write_fake_3w_file(
        ds / "0" / "WELL-A_normal.parquet",
        labels=[0, 0, 0, 0, 0],
    )
    # Event dir "2" — normal → transient → fault.
    _write_fake_3w_file(
        ds / "2" / "WELL-B_dhsv.parquet",
        labels=[0, 0, 102, 2, 2],
    )
    # Event dir "2" — contains a foreign label that must be filtered out.
    _write_fake_3w_file(
        ds / "2" / "WELL-C_mixed.parquet",
        labels=[0, 5, 5, 102, 2],   # 5s should be dropped
    )
    return ds


class TestPrepareInstance:
    def test_filters_to_target_labels(
        self, fake_dataset: Path, tmp_path: Path
    ) -> None:
        out_root = tmp_path / "processed"
        src = fake_dataset / "2" / "WELL-C_mixed.parquet"

        report = prepare_instance(
            src, event_dir="2", out_root=out_root, validate=True
        )

        assert not report.skipped, report.skip_reason
        assert report.n_rows_in == 5
        assert report.n_rows_out == 3   # only {0, 102, 2} survive
        assert report.out_path is not None
        assert report.out_path.exists()

        out = pd.read_parquet(report.out_path)
        assert set(out[LABEL_COL].unique()) <= {0, 2, 102}
        assert (out["instance_id"] == "WELL-C_mixed").all()
        assert (out["source_event_dir"] == "2").all()

    def test_coerces_int64_label_to_nullable(
        self, tmp_path: Path
    ) -> None:
        """If a parquet arrives with non-nullable int64, _read_raw must
        upgrade it to pandas Int64 so the rest of the pipeline holds."""
        ds = tmp_path / "dataset" / "2"
        src = ds / "WELL-int64.parquet"
        ds.mkdir(parents=True, exist_ok=True)

        idx = pd.date_range("2024-01-01", periods=4, freq="1s")
        df = pd.DataFrame({s: [1.0, 2.0, 3.0, 4.0] for s in SENSORS}, index=idx)
        # NOTE: pd.array (not pd.Series) so values aren't NaN'd by index align.
        df[LABEL_COL] = pd.array([0, 0, 102, 2], dtype="int64")  # lowercase!
        df.to_parquet(src, engine="pyarrow", compression="brotli", index=True)

        report = prepare_instance(
            src, event_dir="2", out_root=tmp_path / "processed", validate=True
        )

        assert not report.skipped, report.skip_reason
        out = pd.read_parquet(report.out_path)
        assert isinstance(out[LABEL_COL].dtype, pd.Int64Dtype)

    def test_skips_when_all_rows_are_filtered(
        self, tmp_path: Path
    ) -> None:
        ds = tmp_path / "dataset" / "2"
        src = ds / "WELL-D_only_other.parquet"
        _write_fake_3w_file(src, labels=[5, 6, 7])
        out_root = tmp_path / "processed"

        report = prepare_instance(
            src, event_dir="2", out_root=out_root, validate=True
        )

        assert report.skipped
        assert "no rows" in report.skip_reason
        assert report.out_path is None


class TestPrepareDataset:
    def test_processes_all_dirs_and_writes_summary(
        self, fake_dataset: Path, tmp_path: Path
    ) -> None:
        out_root = tmp_path / "processed"

        summary = prepare_dataset(
            event_dirs=("0", "2"),
            dataset_root=fake_dataset,
            out_root=out_root,
            validate=True,
        )

        assert summary.n_instances == 3
        assert summary.n_kept == 3
        assert (out_root / "0" / "WELL-A_normal.parquet").exists()
        assert (out_root / "2" / "WELL-B_dhsv.parquet").exists()
        assert (out_root / "_summary.csv").exists()

        sdf = pd.read_csv(out_root / "_summary.csv")
        assert set(sdf["source_event_dir"].astype(str)) == {"0", "2"}
        assert (sdf["n_rows_out"] > 0).all()

    def test_max_instances_cap(
        self, fake_dataset: Path, tmp_path: Path
    ) -> None:
        summary = prepare_dataset(
            event_dirs=("0", "2"),
            dataset_root=fake_dataset,
            out_root=tmp_path / "processed",
            max_instances=1,
        )
        assert summary.n_instances == 1

    def test_missing_dataset_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            prepare_dataset(
                dataset_root=tmp_path / "does-not-exist",
                out_root=tmp_path / "processed",
            )