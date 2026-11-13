import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from neural_radar.artifacts import package_versions, write_csv, write_json
from neural_radar.evaluation import ScoreSummary, load_summaries, summary_rows
from neural_radar.experiment import DEFAULT_PROTOCOL, DetectionProtocol
from neural_radar.metrics import noise_scores, threshold_for_pfa
from neural_radar.neural import load_checkpoint, receptive_field, score_batch
from neural_radar.neural_data import network_input

REFERENCES = ("ca_cfar", "os_cfar", "ca_cfar_3x3")
SUMMARY_KEYS = (
    "measured_pfa",
    "pd_all_snr",
    "snr_for_pd_0.5_db",
    "snr_for_pd_0.8_db",
    "pd_at_scenario_pfa_1e-05",
    "pd_at_scenario_pfa_0.0001",
    "pd_at_scenario_pfa_0.001",
)
BOOTSTRAP_DRAWS = 2000


def collect(frames, config):
    inputs, cells, snrs = [], [], []
    for frame in frames:
        inputs.append(network_input(frame.power))
        cells.append(frame.scene.target_cells(config))
        snrs.append(frame.scene.integrated_snr_db)
    return np.stack(inputs), cells, snrs


def load_models(protocol: DetectionProtocol, checkpoints: Path, device: torch.device) -> dict:
    models = {}
    for run in protocol.values["neural"]["runs"]:
        for seed in protocol.values["neural"]["seeds"]:
            path = checkpoints / f"{run['name']}_seed{seed}.pt"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing checkpoint {path}; run scripts/train_detector.py first"
                )
            model, settings, payload = load_checkpoint(path, device)
            models[f"{run['name']}_seed{seed}"] = (model, run["name"], seed, payload)
    return models


def neural_thresholds(protocol, models, device) -> dict[str, float]:
    config = protocol.config
    inputs, cells, _ = collect(protocol.calibration_frames(), config)
    thresholds = {}
    for name, (model, *_rest) in models.items():
        scores = score_batch(model, inputs, device)
        noise = np.concatenate(
            [noise_scores(score, c, config) for score, c in zip(scores, cells, strict=True)]
        )
        thresholds[name] = threshold_for_pfa(noise, protocol.false_alarm_rate)
    return thresholds


def score_scenario(protocol, scenario, models, device):
    config = protocol.config
    inputs, cells, snrs = collect(protocol.test_frames(scenario), config)
    summaries = {name: ScoreSummary() for name in models}
    for name, (model, *_rest) in models.items():
        scores = score_batch(model, inputs, device)
        for score, c, snr in zip(scores, cells, snrs, strict=True):
            summaries[name].add(score, c, snr, config)
    frame_of_target = np.repeat(np.arange(len(cells)), [len(c) for c in cells])
    return summaries, frame_of_target


def paired_difference(hits_a, hits_b, frame_of_target, rng) -> tuple[float, float, float]:
    frames = int(frame_of_target.max()) + 1
    difference = np.bincount(
        frame_of_target, weights=hits_a.astype(float) - hits_b.astype(float), minlength=frames
    )
    counts = np.bincount(frame_of_target, minlength=frames).astype(float)
    estimate = difference.sum() / counts.sum()
    draws = rng.integers(0, frames, size=(BOOTSTRAP_DRAWS, frames))
    samples = difference[draws].sum(axis=1) / np.maximum(counts[draws].sum(axis=1), 1.0)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(estimate), float(low), float(high)


def paired_rows(scenario, summaries, thresholds, frame_of_target, models, rate, rng) -> list[dict]:
    rows = []
    for name, (_model, run, seed, _payload) in models.items():
        target, _snr = summaries[name].targets()
        for reference in REFERENCES:
            reference_target, _ = summaries[reference].targets()
            for basis in ("calibrated", "matched"):
                if basis == "calibrated":
                    own = thresholds[name]
                    other = thresholds[reference]
                else:
                    own = summaries[name].threshold_for(rate)
                    other = summaries[reference].threshold_for(rate)
                estimate, low, high = paired_difference(
                    target > own, reference_target > other, frame_of_target, rng
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "run": run,
                        "seed": seed,
                        "reference": reference,
                        "threshold_basis": basis,
                        "pd_difference": estimate,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    return rows


def aggregate(overview_rows: list[dict], models: dict) -> list[dict]:
    run_of = {name: run for name, (_m, run, _s, _p) in models.items()}
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in overview_rows:
        key = (row["scenario"], run_of.get(row["detector"], row["detector"]))
        grouped.setdefault(key, []).append(row)
    result = []
    for (scenario, detector), rows in grouped.items():
        entry = {
            "scenario": scenario,
            "detector": detector,
            "seeds": len(rows),
            "pfa_is_lower_bound": any(bool(r["pfa_is_lower_bound"]) for r in rows),
        }
        for key in SUMMARY_KEYS:
            values = np.array([float(r[key]) for r in rows])
            entry[key] = float(np.mean(values))
            if len(rows) > 1:
                entry[f"{key}_min"] = float(np.min(values))
                entry[f"{key}_max"] = float(np.max(values))
        result.append(entry)
    return result


def aggregate_paired(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault(
            (row["scenario"], row["run"], row["reference"], row["threshold_basis"]), []
        ).append(row)
    result = []
    for (scenario, run, reference, basis), chosen in grouped.items():
        result.append(
            {
                "scenario": scenario,
                "run": run,
                "reference": reference,
                "threshold_basis": basis,
                "pd_difference_mean": float(np.mean([r["pd_difference"] for r in chosen])),
                "pd_difference_min_seed": float(min(r["pd_difference"] for r in chosen)),
                "pd_difference_max_seed": float(max(r["pd_difference"] for r in chosen)),
                "seeds_with_ci_above_zero": sum(r["ci_low"] > 0 for r in chosen),
                "seeds_with_ci_below_zero": sum(r["ci_high"] < 0 for r in chosen),
                "seeds": len(chosen),
            }
        )
    return result


def curve_series(curve_rows, scenario, detectors):
    series = {}
    for detector in detectors:
        chosen = [r for r in curve_rows if r["scenario"] == scenario and r["group"] == detector]
        levels = sorted({r["snr_db"] for r in chosen})
        values = np.array([[r["pd"] for r in chosen if r["snr_db"] == level] for level in levels])
        if values.size:
            series[detector] = (np.array(levels), values)
    return series


def plot_curves(curve_rows, scenarios, path: Path, title: str) -> None:
    detectors = ["oracle", "ca_cfar", "os_cfar", "ca_cfar_3x3", "cnn_local", "cnn_unet"]
    styles = {
        "oracle": ("--", "0.4"),
        "ca_cfar": ("-", "C0"),
        "os_cfar": ("-", "C1"),
        "ca_cfar_3x3": ("-.", "C4"),
        "cnn_local": ("-", "C2"),
        "cnn_unet": ("-", "C3"),
    }
    columns = 3
    rows = int(np.ceil(len(scenarios) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.3 * rows), squeeze=False)
    for axis, scenario in zip(axes.ravel(), scenarios, strict=False):
        for detector, (levels, values) in curve_series(curve_rows, scenario, detectors).items():
            line, color = styles[detector]
            axis.plot(
                levels,
                values.mean(axis=1),
                line,
                color=color,
                marker="o",
                markersize=3,
                label=detector,
            )
            if values.shape[1] > 1:
                axis.fill_between(
                    levels, values.min(axis=1), values.max(axis=1), color=color, alpha=0.2
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


def plot_false_alarms(aggregated, scenarios, design_rate, path: Path) -> None:
    detectors = [
        "oracle",
        "ca_cfar",
        "go_cfar",
        "os_cfar",
        "ca_cfar_3x3",
        "cnn_local",
        "cnn_unet",
    ]
    figure, axis = plt.subplots(figsize=(9, 3.8))
    width = 0.8 / len(detectors)
    for index, detector in enumerate(detectors):
        values = []
        for scenario in scenarios:
            match = [
                r for r in aggregated if r["scenario"] == scenario and r["detector"] == detector
            ]
            values.append(match[0]["measured_pfa"] if match else np.nan)
        positions = np.arange(len(scenarios)) + (index - (len(detectors) - 1) / 2) * width
        axis.bar(positions, np.maximum(values, 1e-7), width=width, label=detector)
    axis.axhline(design_rate, color="black", linestyle=":", linewidth=1, label="design Pfa")
    axis.set_yscale("log")
    axis.set_xticks(np.arange(len(scenarios)), scenarios, fontsize=8)
    axis.set_ylabel("Measured false-alarm rate")
    axis.set_title("False alarms per cell with thresholds fixed on homogeneous noise", fontsize=9)
    axis.legend(fontsize=7, ncol=3)
    axis.grid(alpha=0.3, axis="y")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--checkpoints", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    protocol = DetectionProtocol.load(arguments.protocol)
    checkpoints = arguments.checkpoints or protocol.path("checkpoints")
    output = arguments.output or protocol.path("neural_results")
    output.mkdir(parents=True, exist_ok=True)
    datasets = protocol.path("datasets")
    classical = load_summaries(datasets / "classical_scores.npz")
    thresholds = json.loads((datasets / "classical_thresholds.json").read_text(encoding="ascii"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(protocol, checkpoints, device)
    start = time.perf_counter()
    thresholds.update(neural_thresholds(protocol, models, device))
    print(
        "neural thresholds",
        {k: round(v, 3) for k, v in thresholds.items()},
        f"{time.perf_counter() - start:.0f}s",
        flush=True,
    )
    rate = protocol.false_alarm_rate
    rng = np.random.default_rng(protocol.values["test"]["seed"])
    curve_rows, overview_rows, paired = [], [], []
    for scenario in protocol.scenarios:
        start = time.perf_counter()
        summaries, frame_of_target = score_scenario(protocol, scenario, models, device)
        reference_count = classical[scenario]["oracle"].targets()[0].size
        if reference_count != frame_of_target.size:
            raise RuntimeError(
                f"{scenario}: regenerated frames do not match the classical benchmark"
            )
        summaries = {**classical[scenario], **summaries}
        curves, overview = summary_rows(
            scenario, summaries, thresholds, protocol.values["roc_false_alarm_rates"]
        )
        run_of = {name: run for name, (_m, run, _s, _p) in models.items()}
        for row in curves:
            row["group"] = run_of.get(row["detector"], row["detector"])
        curve_rows += curves
        overview_rows += overview
        paired += paired_rows(scenario, summaries, thresholds, frame_of_target, models, rate, rng)
        print(f"{scenario} done in {time.perf_counter() - start:.0f}s", flush=True)
    aggregated = aggregate(overview_rows, models)
    paired_summary = aggregate_paired(paired)
    for row in aggregated:
        print(
            f"{row['scenario']:>13} {row['detector']:>9} "
            f"Pfa={'>=' if row['pfa_is_lower_bound'] else ''}{row['measured_pfa']:.3g} "
            f"SNR@Pd0.5={row['snr_for_pd_0.5_db']:.2f} SNR@Pd0.8={row['snr_for_pd_0.8_db']:.2f} "
            f"Pd@own1e-4={row['pd_at_scenario_pfa_0.0001']:.3f}"
        )
    write_csv(curve_rows, output / "pd_curves.csv")
    write_csv(overview_rows, output / "overview_per_seed.csv")
    write_csv(aggregated, output / "overview.csv")
    write_csv(paired, output / "paired_differences_per_seed.csv")
    write_csv(paired_summary, output / "paired_differences.csv")
    plot_curves(
        curve_rows,
        protocol.scenarios,
        output / "pd_curves.png",
        f"CNN (mean, band = 3-seed range) vs CFAR, thresholds set for Pfa = {rate:g} on homogeneous noise",
    )
    plot_false_alarms(aggregated, protocol.scenarios, rate, output / "false_alarm_rates.png")
    write_json(
        {
            "protocol": protocol.values,
            "thresholds": thresholds,
            "checkpoints": {
                name: {
                    "run": run,
                    "seed": seed,
                    "parameters": payload["parameters"],
                    "best_epoch": payload["best_epoch"],
                    "receptive_field": receptive_field(payload["settings"]["architecture"]),
                }
                for name, (_m, run, seed, payload) in models.items()
            },
            "bootstrap": f"{BOOTSTRAP_DRAWS} frame-level resamples, paired on identical frames",
            "device": str(device),
            "versions": package_versions(),
        },
        output / "metadata.json",
    )


if __name__ == "__main__":
    main()
