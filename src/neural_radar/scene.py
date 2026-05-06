from dataclasses import dataclass, field

import numpy as np

from neural_radar.config import RadarConfig


@dataclass(frozen=True)
class Target:
    range_bin: float
    doppler_bin: float
    snr_in_db: float
    phase_rad: float = 0.0

    def nearest_cell(self, config: RadarConfig) -> tuple[int, int]:
        row = int(round(self.range_bin)) % config.samples_per_chirp
        column = (int(round(self.doppler_bin)) + config.chirps // 2) % config.chirps
        return row, column


@dataclass(frozen=True)
class Scene:
    targets: tuple[Target, ...]
    noise_profile: np.ndarray
    interference_chirps: tuple[int, ...] = ()
    interference_start: int = 0
    interference_length: int = 0
    interference_inr_db: float = 0.0
    kind: str = "homogeneous"
    integrated_snr_db: float = float("nan")
    metadata: dict = field(default_factory=dict)

    def target_cells(self, config: RadarConfig) -> np.ndarray:
        if not self.targets:
            return np.zeros((0, 2), dtype=np.int64)
        return np.array([t.nearest_cell(config) for t in self.targets], dtype=np.int64)


def white_profile(config: RadarConfig) -> np.ndarray:
    return np.ones(config.samples_per_chirp)


def edge_profile(config: RadarConfig, edge_bin: int, clutter_to_noise_db: float) -> np.ndarray:
    profile = np.ones(config.samples_per_chirp)
    profile[:edge_bin] += 10.0 ** (clutter_to_noise_db / 10.0)
    return profile


def k_texture_profile(config: RadarConfig, shape: float, rng: np.random.Generator) -> np.ndarray:
    return rng.gamma(shape, 1.0 / shape, size=config.samples_per_chirp)


def per_sample_snr_db(integrated_snr_db: float, config: RadarConfig) -> float:
    return integrated_snr_db - config.coherent_gain_db


def target_amplitude_snr_db(
    mean_integrated_snr_db: float, swerling: int, rng: np.random.Generator
) -> float:
    if swerling == 0:
        return mean_integrated_snr_db
    if swerling == 1:
        mean_linear = 10.0 ** (mean_integrated_snr_db / 10.0)
        return 10.0 * np.log10(rng.exponential(mean_linear))
    raise ValueError(f"Unsupported Swerling case: {swerling}")


def random_position(
    config: RadarConfig,
    rng: np.random.Generator,
    low_range: float | None = None,
    high_range: float | None = None,
) -> tuple[float, float]:
    margin = config.range_margin + 2
    low = margin if low_range is None else max(margin, low_range)
    high = (
        config.samples_per_chirp - margin
        if high_range is None
        else min(config.samples_per_chirp - margin, high_range)
    )
    return float(rng.uniform(low, high)), float(rng.uniform(-config.chirps / 2, config.chirps / 2))


def separated(
    candidate: tuple[float, float],
    placed: list[tuple[float, float]],
    config: RadarConfig,
    minimum: float,
) -> bool:
    for range_bin, doppler_bin in placed:
        doppler_gap = abs(candidate[1] - doppler_bin) % config.chirps
        doppler_gap = min(doppler_gap, config.chirps - doppler_gap)
        if max(abs(candidate[0] - range_bin), doppler_gap) < minimum:
            return False
    return True


def place_targets(
    count: int,
    config: RadarConfig,
    rng: np.random.Generator,
    cluster_fraction: float = 0.0,
    low_range: float | None = None,
    high_range: float | None = None,
    minimum_separation: float = 3.0,
) -> list[tuple[float, float]]:
    placed: list[tuple[float, float]] = []
    attempts = 0
    while len(placed) < count and attempts < 1000:
        attempts += 1
        if placed and rng.random() < cluster_fraction:
            anchor = placed[int(rng.integers(len(placed)))]
            candidate = (
                anchor[0] + rng.uniform(3.0, 6.0) * rng.choice([-1.0, 1.0]),
                anchor[1] + rng.uniform(-3.0, 3.0),
            )
            margin = config.range_margin + 2
            if not margin <= candidate[0] < config.samples_per_chirp - margin:
                continue
            if low_range is not None and candidate[0] < low_range:
                continue
            candidate = (
                candidate[0],
                (candidate[1] + config.chirps / 2) % config.chirps - config.chirps / 2,
            )
        else:
            candidate = random_position(config, rng, low_range, high_range)
        if separated(candidate, placed, config, minimum_separation):
            placed.append(candidate)
    return placed


def build_scene(
    kind: str,
    integrated_snr_db: float,
    config: RadarConfig,
    rng: np.random.Generator,
    settings: dict,
) -> Scene:
    count_low, count_high = settings.get("targets", [1, 4])
    count = int(rng.integers(count_low, count_high + 1))
    swerling = int(settings.get("swerling", 1))
    profile = white_profile(config)
    low_range = None
    metadata: dict = {}
    interference = {}
    if kind == "clutter_edge":
        edge = int(rng.integers(*settings["edge_range"]))
        profile = edge_profile(config, edge, settings["clutter_to_noise_db"])
        low_range = edge + 2.0
        metadata["edge_bin"] = edge
    elif kind == "k_clutter":
        profile = k_texture_profile(config, settings["k_shape"], rng)
    elif kind == "interference":
        chirps = rng.choice(config.chirps, size=settings["interference_chirps"], replace=False)
        length = settings["interference_length"]
        interference = {
            "interference_chirps": tuple(sorted(int(c) for c in chirps)),
            "interference_start": int(rng.integers(0, config.samples_per_chirp - length)),
            "interference_length": length,
            "interference_inr_db": settings["interference_inr_db"],
        }
    elif kind not in ("homogeneous", "dense"):
        raise ValueError(f"Unknown scene kind: {kind}")
    positions = place_targets(
        count, config, rng, settings.get("cluster_fraction", 0.0), low_range=low_range
    )
    targets = tuple(
        Target(
            range_bin=position[0],
            doppler_bin=position[1],
            snr_in_db=per_sample_snr_db(
                target_amplitude_snr_db(integrated_snr_db, swerling, rng), config
            ),
            phase_rad=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
        for position in positions
    )
    return Scene(
        targets=targets,
        noise_profile=profile,
        kind=kind,
        integrated_snr_db=integrated_snr_db,
        metadata=metadata,
        **interference,
    )
