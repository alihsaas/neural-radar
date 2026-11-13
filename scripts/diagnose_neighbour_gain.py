import argparse
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from neural_radar.artifacts import write_csv
from neural_radar.cfar import box_sum
from neural_radar.experiment import DEFAULT_PROTOCOL, DetectionProtocol
from neural_radar.metrics import (
    noise_scores,
    target_scores,
    threshold_for_pfa,
    valid_rows,
    wilson_interval,
)
from neural_radar.neural import load_checkpoint, score_batch
from neural_radar.neural_data import network_input
from neural_radar.simulation import generate_frames


def peak_only(score: np.ndarray) -> np.ndarray:
    return np.where(score >= ndimage.maximum_filter(score, size=3, mode="wrap"), score, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    arguments = parser.parse_args()
    protocol = DetectionProtocol.load(arguments.protocol)
    config = protocol.config
    diagnostics = protocol.values["diagnostics"]
    settings = protocol.values["scenarios"]["homogeneous"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = protocol.path("checkpoints") / f"{diagnostics['checkpoint']}.pt"
    model, _settings, _payload = load_checkpoint(checkpoint, device)
    scorers = {
        "known_noise_single_cell": lambda power: power,
        "known_noise_peak_only": peak_only,
        "known_noise_sum_2x2": lambda power: box_sum(power, 2),
        "known_noise_sum_3x3": lambda power: box_sum(power, 3),
    }

    def all_scores(frames):
        maps = {name: [scorer(f.power) for f in frames] for name, scorer in scorers.items()}
        inputs = np.stack([network_input(f.power) for f in frames])
        maps[diagnostics["checkpoint"]] = list(score_batch(model, inputs, device))
        return maps

    empty = list(
        generate_frames(
            "homogeneous",
            [float("nan")],
            diagnostics["empty_frames"],
            dict(settings, targets=[0, 0]),
            config,
            diagnostics["seed"],
            0,
            protocol.values["window"],
        )
    )
    empty_maps = all_scores(empty)
    no_cells = np.zeros((0, 2), dtype=np.int64)
    thresholds = {}
    rows = []
    for name, maps in empty_maps.items():
        noise = np.concatenate([noise_scores(m, no_cells, config) for m in maps])
        thresholds[name] = threshold_for_pfa(noise, protocol.false_alarm_rate)
        exceed = [m[valid_rows(config)] > thresholds[name] for m in maps]
        rows.append(
            {
                "detector": name,
                "snr_db": "",
                "threshold": thresholds[name],
                "false_alarm_cells": int(sum(e.sum() for e in exceed)),
                "false_alarm_clusters": int(sum(ndimage.label(e)[1] for e in exceed)),
            }
        )
    for snr in diagnostics["snr_db"]:
        frames = list(
            generate_frames(
                "homogeneous",
                [float(snr)],
                diagnostics["frames_per_snr"],
                settings,
                config,
                diagnostics["seed"],
                1 + int(snr),
                protocol.values["window"],
            )
        )
        cells = [f.scene.target_cells(config) for f in frames]
        for name, maps in all_scores(frames).items():
            target = np.concatenate(
                [target_scores(m, c, config) for m, c in zip(maps, cells, strict=True)]
            )
            hits = int(np.sum(target > thresholds[name]))
            low, high = wilson_interval(hits, target.size)
            rows.append(
                {
                    "detector": name,
                    "snr_db": snr,
                    "targets": int(target.size),
                    "pd": hits / target.size,
                    "pd_ci_low": low,
                    "pd_ci_high": high,
                }
            )
            print(f"snr={snr} {name:>26} Pd={hits / target.size:.3f} [{low:.3f}, {high:.3f}]")
    for row in rows[: len(empty_maps)]:
        print(
            f"{row['detector']:>26} false alarm cells={row['false_alarm_cells']} "
            f"clusters={row['false_alarm_clusters']}"
        )
    write_csv(rows, protocol.path("neural_results") / "neighbour_gain_diagnostic.csv")


if __name__ == "__main__":
    main()
