from __future__ import annotations

import pandera as pa
from pandera import Column, DataFrameSchema, Check


# =========================
# SCHEMA DOS DADOS BRUTOS C-MAPSS
# =========================
# 3 settings + 21 sensores + unit_id + cycle
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
SETTING_COLUMNS = [f"setting_{i}" for i in range(1, 4)]


def _sensor_columns_schema() -> dict:
    return {
        col: Column(float, nullable=False, coerce=True)
        for col in SENSOR_COLUMNS
    }


def _setting_columns_schema() -> dict:
    return {
        col: Column(float, nullable=False, coerce=True)
        for col in SETTING_COLUMNS
    }


# Schema dos dados brutos: depois de prepare.py mas antes de feature engineering
RAW_PROCESSED_SCHEMA = DataFrameSchema(
    {
        "unit_id": Column(int, Check.greater_than(0), nullable=False, coerce=True),
        "cycle": Column(int, Check.greater_than(0), nullable=False, coerce=True),
        "rul": Column(int, Check.greater_than_or_equal_to(0), nullable=False, coerce=True),
        **_setting_columns_schema(),
        **_sensor_columns_schema(),
    },
    strict=False,  # permite colunas extras (mas valida as obrigatórias)
    coerce=True,
)


# Schema dos dados COM features (após build_features.py)
# Aqui não validamos cada feature derivada — só garantimos que as colunas obrigatórias
# continuam saudáveis e que não há NaN no target.
FEATURE_SCHEMA = DataFrameSchema(
    {
        "unit_id": Column(int, Check.greater_than(0), nullable=False, coerce=True),
        "cycle": Column(int, Check.greater_than(0), nullable=False, coerce=True),
        "rul": Column(int, Check.greater_than_or_equal_to(0), nullable=False, coerce=True),
    },
    strict=False,
    coerce=True,
)


def validate_raw_processed(df, name: str = "data"):
    """Valida dados brutos processados (saída do prepare.py)."""
    try:
        return RAW_PROCESSED_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        raise ValueError(f"[Pandera] Validação falhou para {name}:\n{e.failure_cases}") from e


def validate_features(df, name: str = "features"):
    """Valida dados com features (saída do build_features.py)."""
    try:
        return FEATURE_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        raise ValueError(f"[Pandera] Validação falhou para {name}:\n{e.failure_cases}") from e