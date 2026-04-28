from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.serving.inference import metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a sample /explain request payload")
    parser.add_argument("--out", type=str, default="test_explain_payload.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--normalized", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    seq_len = int(metadata["seq_len"]) if metadata["seq_len"] is not None else 50
    feature_dim = int(metadata["feature_dim"])

    seq = np.random.randn(seq_len, feature_dim).tolist()
    payload = {
        "sequence": seq,
        "normalized": bool(args.normalized),
        "top_k": int(args.top_k),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(args.out)


if __name__ == "__main__":
    main()
