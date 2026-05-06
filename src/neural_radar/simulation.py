from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from neural_radar.config import RadarConfig
from neural_radar.scene import Scene, build_scene
from neural_radar.signal import simulate_map

SCENE_KINDS = ("homogeneous", "dense", "clutter_edge", "k_clutter", "interference")


@dataclass(frozen=True)
class Frame:
    scene: Scene
    power: np.ndarray


def frame_rng(root_seed: int, stream: int, index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([root_seed, stream, index]))


def generate_frames(
    kind: str,
    snr_values_db: list[float],
    frames_per_snr: int,
    settings: dict,
    config: RadarConfig,
    root_seed: int,
    stream: int,
    window: str = "hann",
) -> Iterator[Frame]:
    index = 0
    for snr_db in snr_values_db:
        for _ in range(frames_per_snr):
            rng = frame_rng(root_seed, stream, index)
            scene = build_scene(kind, snr_db, config, rng, settings)
            yield Frame(scene, simulate_map(scene, config, rng, window))
            index += 1


def mixture_frames(
    count: int,
    mixture: dict[str, float],
    snr_range_db: tuple[float, float],
    empty_fraction: float,
    scenario_settings: dict[str, dict],
    config: RadarConfig,
    root_seed: int,
    stream: int,
) -> Iterator[Frame]:
    kinds = list(mixture)
    weights = np.array([mixture[k] for k in kinds], dtype=np.float64)
    weights /= weights.sum()
    for index in range(count):
        rng = frame_rng(root_seed, stream, index)
        kind = kinds[int(rng.choice(len(kinds), p=weights))]
        settings = dict(scenario_settings[kind])
        if rng.random() < empty_fraction:
            settings["targets"] = [0, 0]
        snr_db = float(rng.uniform(*snr_range_db))
        scene = build_scene(kind, snr_db, config, rng, settings)
        yield Frame(scene, simulate_map(scene, config, rng))
