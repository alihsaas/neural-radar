import numpy as np
import pytest

from neural_radar.theory import pd_swerling0, pd_swerling1, threshold_known_noise


def test_zero_snr_detection_equals_false_alarm_rate():
    assert pd_swerling0(0.0, 1e-4) == pytest.approx(1e-4, rel=1e-6)
    assert pd_swerling1(0.0, 1e-4) == pytest.approx(1e-4, rel=1e-9)


def test_swerling0_matches_monte_carlo(rng):
    snr = 10.0
    samples = 200_000
    noise = (rng.standard_normal(samples) + 1j * rng.standard_normal(samples)) / np.sqrt(2)
    power = np.abs(np.sqrt(snr) + noise) ** 2
    measured = np.mean(power > threshold_known_noise(1e-3))
    assert measured == pytest.approx(float(pd_swerling0(snr, 1e-3)), abs=0.005)


def test_fluctuating_target_needs_more_snr_at_high_pd():
    snr = 10 ** (15 / 10)
    assert pd_swerling1(snr, 1e-4) < pd_swerling0(snr, 1e-4)
