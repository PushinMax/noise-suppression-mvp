from __future__ import annotations

import numpy as np

from noise_suppression.metrics import si_sdr


def test_si_sdr_prefers_cleaner_signal() -> None:
    reference = np.array([0.0, 0.5, -0.5, 0.25, -0.25], dtype=np.float32)
    close_estimate = reference + np.array([0.0, 0.01, -0.01, 0.01, -0.01], dtype=np.float32)
    noisy_estimate = reference + np.array([0.1, -0.2, 0.2, -0.1, 0.1], dtype=np.float32)

    assert si_sdr(reference, close_estimate) > si_sdr(reference, noisy_estimate)
