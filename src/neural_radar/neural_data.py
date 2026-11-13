from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neural_radar.config import RadarConfig
from neural_radar.metrics import EXCLUSION_RADIUS, chebyshev_distance_map, valid_rows
from neural_radar.simulation import Frame, mixture_frames


def network_input(power: np.ndarray) -> np.ndarray:
    level_db = 10.0 * np.log10(np.maximum(power, 1e-12))
    return ((level_db - np.median(level_db)) / 10.0).astype(np.float32)


def label_maps(frame: Frame, config: RadarConfig) -> tuple[np.ndarray, np.ndarray]:
    cells = frame.scene.target_cells(config)
    label = np.zeros((config.samples_per_chirp, config.chirps), dtype=np.uint8)
    weight = np.zeros_like(label)
    weight[valid_rows(config)] = 1
    if len(cells):
        near = chebyshev_distance_map(cells, config) <= EXCLUSION_RADIUS
        weight[near] = 0
        label[cells[:, 0], cells[:, 1]] = 1
        weight[cells[:, 0], cells[:, 1]] = 1
    return label, weight


@dataclass(frozen=True)
class DetectionDataset:
    inputs: np.ndarray
    labels: np.ndarray
    weights: np.ndarray

    def __len__(self) -> int:
        return int(self.inputs.shape[0])


def build_dataset(frames, count: int, config: RadarConfig) -> DetectionDataset:
    shape = (count, config.samples_per_chirp, config.chirps)
    inputs = np.zeros(shape, dtype=np.float16)
    labels = np.zeros(shape, dtype=np.uint8)
    weights = np.zeros(shape, dtype=np.uint8)
    for index, frame in enumerate(frames):
        inputs[index] = network_input(frame.power)
        labels[index], weights[index] = label_maps(frame, config)
    return DetectionDataset(inputs, labels, weights)


def training_datasets(protocol, cache_dir: Path) -> tuple[DetectionDataset, DetectionDataset]:
    training = protocol.values["training"]
    settings = protocol.values["scenarios"]
    config = protocol.config
    result = []
    for stream, (split, count) in enumerate(
        (("train", training["train_frames"]), ("validation", training["validation_frames"]))
    ):
        path = Path(cache_dir) / f"{split}_{training['seed']}_{count}.npz"
        if path.exists():
            with np.load(path) as stored:
                result.append(
                    DetectionDataset(stored["inputs"], stored["labels"], stored["weights"])
                )
            continue
        frames = mixture_frames(
            count,
            training["mixture"],
            tuple(training["snr_range_db"]),
            training["empty_fraction"],
            settings,
            config,
            training["seed"],
            stream,
        )
        dataset = build_dataset(frames, count, config)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, inputs=dataset.inputs, labels=dataset.labels, weights=dataset.weights)
        result.append(dataset)
    return result[0], result[1]
