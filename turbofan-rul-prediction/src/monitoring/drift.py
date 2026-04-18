from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

from src.core.settings import (
    FEATURE_TRAIN_FILE,
    FEATURE_TEST_FILE,
    DROP_COLUMNS,
    REPORTS_DIR,
    ensure_directories,
)


def generate_drift_report(
    reference_path: str = str(FEATURE_TRAIN_FILE),
    current_path: str = str(FEATURE_TEST_FILE),
    output_dir: str = str(REPORTS_DIR),
) -> dict:
    """
    Gera relatório de data drift + target drift comparando
    dados de referência (train) com dados atuais (test/produção).

    Produz:
    - HTML interativo (pra humanos)
    - JSON com resultados (pra automação / alertas)
    """
    ensure_directories()

    ref_df = pd.read_parquet(reference_path)
    cur_df = pd.read_parquet(current_path)

    # Remove colunas não-features
    feature_cols = [c for c in ref_df.columns if c not in DROP_COLUMNS]

    ref_features = ref_df[feature_cols].copy()
    cur_features = cur_df[feature_cols].copy()

    # Adiciona o target pra análise de target drift
    ref_features["rul"] = ref_df["rul"].values
    cur_features["rul"] = cur_df["rul"].values

    column_mapping = ColumnMapping(
        target="rul",
        numerical_features=feature_cols,
    )

    # Gera report com data drift + target drift
    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
    ])

    report.run(
        reference_data=ref_features,
        current_data=cur_features,
        column_mapping=column_mapping,
    )

    # Salva HTML interativo
    output_path = Path(output_dir)
    html_path = output_path / "drift_report.html"
    json_path = output_path / "drift_report.json"

    report.save_html(str(html_path))
    report.save_json(str(json_path))

    # Extrai resumo pra print e automação
    report_dict = report.as_dict()

    # Conta features com drift detectado
    drift_results = []
    for metric_result in report_dict.get("metrics", []):
        result = metric_result.get("result", {})
        if "drift_by_columns" in result:
            for col_name, col_data in result["drift_by_columns"].items():
                if col_data.get("drift_detected", False):
                    drift_results.append({
                        "feature": col_name,
                        "drift_score": col_data.get("drift_score", 0),
                        "stattest": col_data.get("stattest_name", ""),
                    })

    summary = {
        "n_features_analyzed": len(feature_cols),
        "n_features_drifted": len(drift_results),
        "drift_ratio": len(drift_results) / max(len(feature_cols), 1),
        "drifted_features": drift_results[:10],  # top 10
        "html_report": str(html_path),
        "json_report": str(json_path),
    }

    return summary


def main():
    print("Generating Evidently drift report...")
    summary = generate_drift_report()

    print(f"\nFeatures analyzed: {summary['n_features_analyzed']}")
    print(f"Features with drift: {summary['n_features_drifted']}")
    print(f"Drift ratio: {summary['drift_ratio']:.2%}")

    if summary["drifted_features"]:
        print("\nTop drifted features:")
        for feat in summary["drifted_features"]:
            print(f"  {feat['feature']}: score={feat['drift_score']:.4f} ({feat['stattest']})")

    print(f"\nHTML report: {summary['html_report']}")
    print(f"JSON report: {summary['json_report']}")

    # Salva summary como JSON separado
    summary_path = Path(summary["json_report"]).parent / "drift_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()