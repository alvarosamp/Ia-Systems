from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import src.data.prepare as prep


def _write_raw_cmapss(file_path: Path, rows: list[list[float]]) -> None:
    # read_cmapss_file usa sep whitespace e header=None
    lines = [" ".join(str(x) for x in row) for row in rows]
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_prepare_main_writes_parquet(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)

    train_raw = raw_dir / "train.txt"
    test_raw = raw_dir / "test.txt"
    rul_raw = raw_dir / "rul.txt"

    train_out = out_dir / "train.parquet"
    test_out = out_dir / "test.parquet"

    # Patch paths inside module
    monkeypatch.setattr(prep, "TRAIN_RAW_FILE", train_raw)
    monkeypatch.setattr(prep, "TEST_RAW_FILE", test_raw)
    monkeypatch.setattr(prep, "RUL_RAW_FILE", rul_raw)
    monkeypatch.setattr(prep, "TRAIN_PROCESSED_FILE", train_out)
    monkeypatch.setattr(prep, "TEST_PROCESSED_FILE", test_out)

    monkeypatch.setattr(prep, "ensure_directories", lambda: out_dir.mkdir(parents=True, exist_ok=True))

    # 26 colunas: unit_id, cycle, setting_1..3, sensor_1..21
    zeros = [0.0] * (3 + 21)

    # Train: duas unidades com ciclos diferentes
    train_rows = [
        [1, 1, *zeros],
        [1, 2, *zeros],
        [1, 3, *zeros],
        [2, 1, *zeros],
        [2, 2, *zeros],
    ]
    test_rows = [
        [1, 1, *zeros],
        [1, 2, *zeros],
        [2, 1, *zeros],
    ]

    _write_raw_cmapss(train_raw, train_rows)
    _write_raw_cmapss(test_raw, test_rows)

    # extra_rul para unit_id 1 e 2
    rul_raw.write_text("5\n7\n", encoding="utf-8")

    prep.main()

    assert train_out.exists()
    assert test_out.exists()

    train_df = pd.read_parquet(train_out)
    test_df = pd.read_parquet(test_out)

    assert "rul" in train_df.columns
    assert "rul" in test_df.columns

    assert train_df["rul"].max() <= prep.RUL_CAP
    assert test_df["rul"].max() <= prep.RUL_CAP


def test_prepare_main_raises_if_missing_train_file(tmp_path, monkeypatch):
    missing = tmp_path / "missing_train.txt"

    monkeypatch.setattr(prep, "TRAIN_RAW_FILE", missing)
    monkeypatch.setattr(prep, "ensure_directories", lambda: None)

    with pytest.raises(FileNotFoundError, match="Arquivo não encontrado"):
        prep.main()
