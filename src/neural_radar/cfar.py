from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage, optimize

from neural_radar.config import RadarConfig


def training_mask(config: RadarConfig) -> np.ndarray:
    half_range = config.guard_range + config.train_range
    half_doppler = config.guard_doppler + config.train_doppler
    mask = np.ones((2 * half_range + 1, 2 * half_doppler + 1), dtype=bool)
    mask[
        config.train_range : config.train_range + 2 * config.guard_range + 1,
        config.train_doppler : config.train_doppler + 2 * config.guard_doppler + 1,
    ] = False
    return mask


def training_cells(config: RadarConfig) -> int:
    return int(training_mask(config).sum())


def wrapped_mean(power: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return ndimage.correlate(power, kernel / kernel.sum(), mode="wrap")


def ca_estimate(power: np.ndarray, config: RadarConfig) -> np.ndarray:
    return wrapped_mean(power, training_mask(config).astype(np.float64))


def go_estimate(power: np.ndarray, config: RadarConfig) -> np.ndarray:
    mask = training_mask(config)
    center = config.guard_range + config.train_range
    lagging = mask.copy()
    leading = mask.copy()
    lagging[center:, :] = False
    leading[: center + 1, :] = False
    return np.maximum(
        wrapped_mean(power, lagging.astype(np.float64)),
        wrapped_mean(power, leading.astype(np.float64)),
    )


def os_estimate(power: np.ndarray, config: RadarConfig, rank_fraction: float = 0.75) -> np.ndarray:
    mask = training_mask(config)
    half_range = config.guard_range + config.train_range
    half_doppler = config.guard_doppler + config.train_doppler
    padded = np.pad(power, ((half_range, half_range), (half_doppler, half_doppler)), mode="wrap")
    windows_view = sliding_window_view(padded, mask.shape)
    cells = windows_view[..., mask]
    rank = int(round(rank_fraction * mask.sum())) - 1
    return np.partition(cells, rank, axis=-1)[..., rank]


def oracle_estimate(power: np.ndarray, config: RadarConfig) -> np.ndarray:
    return np.ones_like(power)


def box_sum(power: np.ndarray, size: int) -> np.ndarray:
    return ndimage.uniform_filter(power, size=size, mode="wrap") * size * size


@dataclass(frozen=True)
class Detector:
    name: str
    estimate: Callable[[np.ndarray, RadarConfig], np.ndarray]
    description: str
    integration: int = 1

    def score(self, power: np.ndarray, config: RadarConfig) -> np.ndarray:
        if self.integration > 1:
            power = box_sum(power, self.integration)
        return power / np.maximum(self.estimate(power, config), 1e-12)


CLASSICAL_DETECTORS = (
    Detector("oracle", oracle_estimate, "Known thermal noise level (not realizable)"),
    Detector("ca_cfar", ca_estimate, "Cell-averaging CFAR, 2-D ring of training cells"),
    Detector("go_cfar", go_estimate, "Greatest-of CFAR, leading/lagging halves in range"),
    Detector("os_cfar", os_estimate, "Ordered-statistic CFAR, 75th percentile of training cells"),
    Detector(
        "ca_cfar_3x3",
        ca_estimate,
        "Cell-averaging CFAR on the 3 x 3 summed map (non-coherent energy integration)",
        integration=3,
    ),
)


def detector(name: str) -> Detector:
    for item in CLASSICAL_DETECTORS:
        if item.name == name:
            return item
    raise KeyError(name)


def ca_alpha_theory(training: int, false_alarm_rate: float) -> float:
    return training * (false_alarm_rate ** (-1.0 / training) - 1.0)


def ca_pfa_theory(alpha: float, training: int) -> float:
    return (1.0 + alpha / training) ** (-training)


def os_pfa_theory(alpha: float, training: int, rank: int) -> float:
    indices = np.arange(rank)
    return float(np.prod((training - indices) / (training - indices + alpha)))


def os_alpha_theory(training: int, rank: int, false_alarm_rate: float) -> float:
    return float(
        optimize.brentq(
            lambda a: np.log(os_pfa_theory(a, training, rank)) - np.log(false_alarm_rate),
            1e-6,
            200.0,
        )
    )
