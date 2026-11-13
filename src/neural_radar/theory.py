import numpy as np
from scipy import stats


def threshold_known_noise(false_alarm_rate: float) -> float:
    return float(-np.log(false_alarm_rate))


def pd_swerling0(snr_linear: np.ndarray | float, false_alarm_rate: float) -> np.ndarray:
    threshold = threshold_known_noise(false_alarm_rate)
    snr = np.asarray(snr_linear, dtype=np.float64)
    return stats.ncx2.sf(2.0 * threshold, 2, 2.0 * snr)


def pd_swerling1(snr_linear: np.ndarray | float, false_alarm_rate: float) -> np.ndarray:
    threshold = threshold_known_noise(false_alarm_rate)
    snr = np.asarray(snr_linear, dtype=np.float64)
    return np.exp(-threshold / (1.0 + snr))
