from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def synthetic_cmapss_df() -> pd.DataFrame:
    """DataFrame pequeno no formato C-MAPSS para testes rápidos."""
    rows = []
    for unit_id in range(1, 6):  # 5 unidades
        n_cycles = 50 + unit_id * 10
        for cycle in range(1, n_cycles + 1):
            row = {"unit_id": unit_id, "cycle": cycle}
            for i in range(1, 4):
                row[f"setting_{i}"] = np.random.randn()
            for i in range(1, 22):
                row[f"sensor_{i}"] = np.random.randn() + cycle * 0.01
            row["rul"] = n_cycles - cycle
            rows.append(row)
    return pd.DataFrame(rows)