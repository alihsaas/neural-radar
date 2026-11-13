import numpy as np
import pytest

from neural_radar.metrics import (
    chebyshev_distance_map,
    noise_scores,
    required_snr_db,
    target_scores,
    threshold_for_pfa,
    wilson_interval,
)


def test_chebyshev_distance_wraps_in_doppler(config):
    distance = chebyshev_distance_map(np.array([[50, 0]]), config)
    assert distance[50, 63] == 1
    assert distance[52, 62] == 2
    assert distance[53, 0] == 3


def test_target_score_takes_best_cell_in_neighbourhood(config):
    score = np.zeros((config.samples_per_chirp, config.chirps))
    score[41, 10] = 5.0
    score[43, 10] = 9.0
    assert target_scores(score, np.array([[40, 10]]), config).tolist() == [5.0]


def test_noise_scores_skip_guard_zone_and_edges(config):
    score = np.ones((config.samples_per_chirp, config.chirps))
    noise = noise_scores(score, np.array([[60, 30]]), config)
    rows = len(config.valid_range_bins)
    assert noise.size == rows * config.chirps - 25


def test_threshold_for_pfa_quantile(rng):
    noise = rng.exponential(1.0, size=2_000_000)
    assert threshold_for_pfa(noise, 1e-3) == pytest.approx(-np.log(1e-3), rel=0.03)


def test_threshold_for_pfa_refuses_small_pools():
    with pytest.raises(ValueError):
        threshold_for_pfa(np.ones(1000), 1e-4)


def test_wilson_interval_contains_estimate():
    low, high = wilson_interval(30, 100)
    assert low < 0.3 < high
    assert wilson_interval(0, 50)[0] == pytest.approx(0.0, abs=1e-12)


def test_required_snr_interpolates():
    snr = np.array([0.0, 10.0, 20.0])
    pd = np.array([0.1, 0.5, 0.9])
    assert required_snr_db(snr, pd, 0.7) == pytest.approx(15.0)
    assert np.isnan(required_snr_db(snr, pd, 0.95))
