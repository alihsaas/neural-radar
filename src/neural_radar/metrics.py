from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from neural_radar.config import RadarConfig

HIT_RADIUS = 1
EXCLUSION_RADIUS = 2


def valid_rows(config: RadarConfig) -> slice:
    return slice(config.valid_range_bins.start, config.valid_range_bins.stop)


def chebyshev_distance_map(cells: np.ndarray, config: RadarConfig) -> np.ndarray:
    rows = np.arange(config.samples_per_chirp)[:, None]
    columns = np.arange(config.chirps)[None, :]
    distance = np.full((config.samples_per_chirp, config.chirps), np.iinfo(np.int64).max // 2)
    for row, column in cells:
        doppler_gap = np.abs(columns - column)
        doppler_gap = np.minimum(doppler_gap, config.chirps - doppler_gap)
        distance = np.minimum(distance, np.maximum(np.abs(rows - row), doppler_gap))
    return distance


def target_scores(score: np.ndarray, cells: np.ndarray, config: RadarConfig) -> np.ndarray:
    size = 2 * HIT_RADIUS + 1
    local_max = ndimage.maximum_filter(score, size=(size, size), mode="wrap")
    return np.array([local_max[row, column] for row, column in cells], dtype=np.float64)


def noise_scores(score: np.ndarray, cells: np.ndarray, config: RadarConfig) -> np.ndarray:
    far = chebyshev_distance_map(cells, config) > EXCLUSION_RADIUS
    rows = valid_rows(config)
    return score[rows][far[rows]]


@dataclass
class ScorePool:
    target: list[np.ndarray]
    noise: list[np.ndarray]
    target_snr_db: list[np.ndarray]

    @classmethod
    def empty(cls) -> "ScorePool":
        return cls([], [], [])

    def add(self, score: np.ndarray, cells: np.ndarray, snr_db: float, config: RadarConfig) -> None:
        self.target.append(target_scores(score, cells, config))
        self.noise.append(noise_scores(score, cells, config))
        self.target_snr_db.append(np.full(len(cells), snr_db))

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.concatenate(self.target) if self.target else np.zeros(0),
            np.concatenate(self.noise) if self.noise else np.zeros(0),
            np.concatenate(self.target_snr_db) if self.target_snr_db else np.zeros(0),
        )


def threshold_for_pfa(noise: np.ndarray, false_alarm_rate: float) -> float:
    if noise.size * false_alarm_rate < 20:
        raise ValueError("Too few noise cells to set this false-alarm rate reliably")
    return float(np.quantile(noise, 1.0 - false_alarm_rate, method="higher"))


def detection_rate(target: np.ndarray, threshold: float) -> float:
    return float(np.mean(target > threshold)) if target.size else float("nan")


def false_alarm_rate(noise: np.ndarray, threshold: float) -> float:
    return float(np.mean(noise > threshold)) if noise.size else float("nan")


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return float("nan"), float("nan")
    p = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    half = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    return float(center - half), float(center + half)


def required_snr_db(snr_db: np.ndarray, detection: np.ndarray, target_pd: float = 0.9) -> float:
    order = np.argsort(snr_db)
    snr_sorted = np.asarray(snr_db)[order]
    pd_sorted = np.maximum.accumulate(np.asarray(detection)[order])
    if pd_sorted[-1] < target_pd:
        return float("nan")
    if pd_sorted[0] >= target_pd:
        return float(snr_sorted[0])
    above = int(np.argmax(pd_sorted >= target_pd))
    x0, x1 = snr_sorted[above - 1], snr_sorted[above]
    y0, y1 = pd_sorted[above - 1], pd_sorted[above]
    return float(x0 + (target_pd - y0) * (x1 - x0) / (y1 - y0))
