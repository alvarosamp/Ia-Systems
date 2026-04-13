from __future__ import annotations

import numpy as np
import pytest

from src.serving.inference import metadata, validate_sequence


def test_validate_sequence_rejects_wrong_seq_len():
    seq_len = int(metadata["seq_len"])
    feature_dim = int(metadata["feature_dim"])

    bad = np.random.randn(seq_len - 1, feature_dim).tolist()

    with pytest.raises(ValueError, match="modelo espera"):
        validate_sequence(bad, normalized=False)


def test_validate_sequence_rejects_non_2d():
    seq_len = int(metadata["seq_len"])
    # Lista 1D com len correta -> passa pelo check de seq_len, falha no check de 2D
    bad = [1.0] * seq_len

    with pytest.raises(ValueError, match="2D"):
        validate_sequence(bad, normalized=False)


def test_validate_sequence_rejects_wrong_feature_dim():
    seq_len = int(metadata["seq_len"])
    feature_dim = int(metadata["feature_dim"])

    bad = np.random.randn(seq_len, feature_dim + 1).tolist()

    with pytest.raises(ValueError, match="features"):
        validate_sequence(bad, normalized=False)
