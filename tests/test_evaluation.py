import numpy as np
import pytest

from neural_radar.evaluation import ScoreSummary, load_summaries, save_summaries
from neural_radar.metrics import noise_scores


def test_tail_keeps_exact_false_alarm_counts(config, rng):
    summary = ScoreSummary()
    pool = []
    cells = np.array([[60, 30]])
    for _ in range(30):
        score = rng.exponential(1.0, size=(config.samples_per_chirp, config.chirps))
        summary.add(score, cells, 10.0, config)
        pool.append(noise_scores(score, cells, config))
    pool = np.concatenate(pool)
    assert summary.noise_cells == pool.size
    assert summary.false_alarm_rate(4.0) == pytest.approx(np.mean(pool > 4.0))


def test_summaries_round_trip(tmp_path, config, rng):
    summary = ScoreSummary()
    summary.add(rng.exponential(1.0, size=(128, 64)), np.array([[50, 10], [70, 40]]), 12.0, config)
    path = tmp_path / "scores.npz"
    save_summaries(path, {"homogeneous": {"ca_cfar": summary}})
    loaded = load_summaries(path)["homogeneous"]["ca_cfar"]
    assert np.array_equal(loaded.targets()[0], summary.targets()[0])
    assert np.array_equal(loaded.noise_tail, summary.noise_tail)
    assert loaded.noise_cells == summary.noise_cells
