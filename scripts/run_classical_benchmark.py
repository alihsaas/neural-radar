import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neural_radar.artifacts import package_versions, write_csv, write_json
from neural_radar.cfar import CLASSICAL_DETECTORS, training_cells
from neural_radar.evaluation import calibrate, save_summaries, score_frames, summary_rows
from neural_radar.experiment import DEFAULT_PROTOCOL, DetectionProtocol, classical_scorers
from neural_radar.scene import Scene, Target, edge_profile, per_sample_snr_db
from neural_radar.signal import simulate_map
from neural_radar.simulation import frame_rng


def plot_curves(curve_rows: list[dict], scenarios: list[str], path: Path, title: str) -> None:
    columns = 3
    rows = int(np.ceil(len(scenarios) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.3 * rows), squeeze=False)
    detectors = list(dict.fromkeys(row["detector"] for row in curve_rows))
    for axis, scenario in zip(axes.ravel(), scenarios, strict=False):
        for name in detectors:
            chosen = [r for r in curve_rows if r["scenario"] == scenario and r["detector"] == name]
            if not chosen:
                continue
            axis.plot(
                [r["snr_db"] for r in chosen],
                [r["pd"] for r in chosen],
                "--" if name in ("oracle",) else "-",
                marker="o",
                markersize=3,
                label=name,
            )
        axis.set_title(scenario, fontsize=9)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.3)
        axis.set_xlabel("Mean integrated SNR (dB)", fontsize=8)
        axis.set_ylabel("Pd", fontsize=8)
    for axis in axes.ravel()[len(scenarios) :]:
        axis.axis("off")
    axes.ravel()[0].legend(fontsize=7)
    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_examples(protocol: DetectionProtocol, path: Path) -> None:
    config = protocol.config
    rng = frame_rng(9, 9, 0)
    targets = (
        Target(60.0, 5.3, per_sample_snr_db(24.0, config), 0.3),
        Target(64.0, 6.1, per_sample_snr_db(22.0, config), 1.1),
        Target(95.5, -12.4, per_sample_snr_db(20.0, config), 2.0),
    )
    scene = Scene(targets, edge_profile(config, 45, 15.0))
    power = simulate_map(scene, config, rng)
    figure, axis = plt.subplots(figsize=(6, 5))
    extent = [
        -config.max_velocity_mps,
        config.max_velocity_mps,
        config.max_range_m,
        0.0,
    ]
    image = axis.imshow(
        10 * np.log10(power), aspect="auto", extent=extent, cmap="viridis", vmin=-10, vmax=30
    )
    for target in targets:
        axis.plot(
            target.doppler_bin * config.velocity_resolution_mps,
            target.range_bin * config.range_resolution_m,
            "r+",
            markersize=10,
        )
    axis.axhline(45 * config.range_resolution_m, color="white", linestyle=":", linewidth=1)
    axis.set_xlabel("Radial velocity (m/s)")
    axis.set_ylabel("Range (m)")
    axis.set_title(
        "Range-Doppler map: 2 close targets, 1 far target, clutter edge at 45 m", fontsize=8
    )
    figure.colorbar(image, ax=axis, label="Power / thermal noise (dB)")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    arguments = parser.parse_args()
    protocol = DetectionProtocol.load(arguments.protocol)
    config = protocol.config
    output = protocol.path("classical_results")
    output.mkdir(parents=True, exist_ok=True)
    scorers = classical_scorers(config)
    start = time.perf_counter()
    thresholds = calibrate(
        protocol.calibration_frames(), scorers, config, protocol.false_alarm_rate
    )
    print("thresholds", thresholds, f"{time.perf_counter() - start:.0f}s")
    curve_rows: list[dict] = []
    overview_rows: list[dict] = []
    all_summaries = {}
    for scenario in protocol.scenarios:
        start = time.perf_counter()
        summaries = score_frames(protocol.test_frames(scenario), scorers, config)
        all_summaries[scenario] = summaries
        curves, overview = summary_rows(
            scenario, summaries, thresholds, protocol.values["roc_false_alarm_rates"]
        )
        curve_rows += curves
        overview_rows += overview
        for row in overview:
            print(
                f"{scenario:>13} {row['detector']:>8} Pfa={'>=' if row['pfa_is_lower_bound'] else ''}{row['measured_pfa']:.3g} "
                f"SNR@Pd0.5={row['snr_for_pd_0.5_db']:.2f} SNR@Pd0.8={row['snr_for_pd_0.8_db']:.2f} "
                f"Pd@own1e-4={row['pd_at_scenario_pfa_0.0001']:.3f}"
            )
        print(f"{scenario} done in {time.perf_counter() - start:.0f}s", flush=True)
    save_summaries(protocol.path("datasets") / "classical_scores.npz", all_summaries)
    write_json(thresholds, protocol.path("datasets") / "classical_thresholds.json")
    write_csv(curve_rows, output / "pd_curves.csv")
    write_csv(overview_rows, output / "overview.csv")
    plot_curves(
        curve_rows,
        protocol.scenarios,
        output / "pd_curves.png",
        f"Classical detectors, thresholds set for Pfa = {protocol.false_alarm_rate:g} on homogeneous noise",
    )
    plot_examples(protocol, output / "example_range_doppler.png")
    write_json(
        {
            "protocol": protocol.values,
            "thresholds": thresholds,
            "detectors": {d.name: d.description for d in CLASSICAL_DETECTORS},
            "training_cells": training_cells(config),
            "versions": package_versions(),
        },
        output / "metadata.json",
    )


if __name__ == "__main__":
    main()
