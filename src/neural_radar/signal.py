import numpy as np
from scipy.signal import windows

from neural_radar.config import RadarConfig
from neural_radar.scene import Scene


def complex_gaussian(
    shape: tuple[int, ...], variance: np.ndarray | float, rng: np.random.Generator
) -> np.ndarray:
    scale = np.sqrt(np.asarray(variance) / 2.0)
    return scale * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def target_signal(scene: Scene, config: RadarConfig) -> np.ndarray:
    fast = np.arange(config.samples_per_chirp)[:, None]
    slow = np.arange(config.chirps)[None, :]
    signal = np.zeros((config.samples_per_chirp, config.chirps), dtype=np.complex128)
    for target in scene.targets:
        amplitude = np.sqrt(10.0 ** (target.snr_in_db / 10.0))
        phase = (
            2.0
            * np.pi
            * (
                target.range_bin * fast / config.samples_per_chirp
                + target.doppler_bin * slow / config.chirps
            )
        )
        signal += amplitude * np.exp(1j * (phase + target.phase_rad))
    return signal


def colored_noise(profile: np.ndarray, config: RadarConfig, rng: np.random.Generator) -> np.ndarray:
    spectrum = complex_gaussian((config.samples_per_chirp, config.chirps), profile[:, None], rng)
    return np.fft.ifft(spectrum, axis=0, norm="ortho")


def interference_signal(scene: Scene, config: RadarConfig, rng: np.random.Generator) -> np.ndarray:
    signal = np.zeros((config.samples_per_chirp, config.chirps), dtype=np.complex128)
    if not scene.interference_chirps:
        return signal
    variance = 10.0 ** (scene.interference_inr_db / 10.0)
    start = scene.interference_start
    stop = start + scene.interference_length
    for chirp in scene.interference_chirps:
        signal[start:stop, chirp] = complex_gaussian((stop - start,), variance, rng)
    return signal


def synthesize(scene: Scene, config: RadarConfig, rng: np.random.Generator) -> np.ndarray:
    return (
        target_signal(scene, config)
        + colored_noise(scene.noise_profile, config, rng)
        + interference_signal(scene, config, rng)
    )


def window_pair(config: RadarConfig, kind: str = "hann") -> tuple[np.ndarray, np.ndarray]:
    if kind == "rect":
        return np.ones(config.samples_per_chirp), np.ones(config.chirps)
    if kind == "hann":
        return (
            windows.hann(config.samples_per_chirp, sym=False),
            windows.hann(config.chirps, sym=False),
        )
    raise ValueError(f"Unknown window: {kind}")


def range_doppler_power(
    samples: np.ndarray, config: RadarConfig, window: str = "hann"
) -> np.ndarray:
    range_window, doppler_window = window_pair(config, window)
    weighted = samples * range_window[:, None] * doppler_window[None, :]
    spectrum = np.fft.fft(np.fft.fft(weighted, axis=0), axis=1)
    spectrum = np.fft.fftshift(spectrum, axes=1)
    noise_gain = np.sum(range_window**2) * np.sum(doppler_window**2)
    return (np.abs(spectrum) ** 2 / noise_gain).astype(np.float64)


def peak_gain_db(config: RadarConfig, window: str = "hann") -> float:
    range_window, doppler_window = window_pair(config, window)
    coherent = (np.sum(range_window) * np.sum(doppler_window)) ** 2
    noise = np.sum(range_window**2) * np.sum(doppler_window**2)
    return float(10.0 * np.log10(coherent / noise))


def simulate_map(
    scene: Scene, config: RadarConfig, rng: np.random.Generator, window: str = "hann"
) -> np.ndarray:
    return range_doppler_power(synthesize(scene, config, rng), config, window)
