import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neural_radar.artifacts import package_versions, write_csv, write_json
from neural_radar.cfar import (
    ca_alpha_theory,
    ca_estimate,
    os_alpha_theory,
    os_estimate,
    training_cells,
)
from neural_radar.config import DEFAULT_CONFIG, PROJECT_ROOT
from neural_radar.metrics import required_snr_db, valid_rows
from neural_radar.scene import Scene, Target, per_sample_snr_db, white_profile
from neural_radar.signal import peak_gain_db, simulate_map
from neural_radar.simulation import frame_rng
from neural_radar.theory import pd_swerling0, pd_swerling1, threshold_known_noise

CONFIG = DEFAULT_CONFIG
ROOT_SEED = 1001


def noise_frames(count: int, window: str, stream: int) -> np.ndarray:
    scene = Scene((), white_profile(CONFIG))
    return np.stack(
        [simulate_map(scene, CONFIG, frame_rng(ROOT_SEED, stream, i), window) for i in range(count)]
    )


def check_noise_normalization() -> list[dict]:
    rows = []
    for stream, window in enumerate(("rect", "hann")):
        maps = noise_frames(300, window, stream)
        rows.append(
            {
                "check": "noise_mean_power",
                "window": window,
                "measured": float(maps.mean()),
                "expected": 1.0,
                "cells": int(maps.size),
            }
        )
    return rows


def check_target_peak() -> list[dict]:
    rows = []
    for window in ("rect", "hann"):
        target = Target(range_bin=40.0, doppler_bin=-7.0, snr_in_db=per_sample_snr_db(20.0, CONFIG))
        scene = Scene((target,), np.zeros(CONFIG.samples_per_chirp))
        power = simulate_map(scene, CONFIG, frame_rng(ROOT_SEED, 9, 0), window)
        peak = np.unravel_index(int(np.argmax(power)), power.shape)
        rows.append(
            {
                "check": "noiseless_peak",
                "window": window,
                "peak_cell": f"{peak[0]},{peak[1]}",
                "expected_cell": f"{target.nearest_cell(CONFIG)[0]},{target.nearest_cell(CONFIG)[1]}",
                "peak_db": float(10 * np.log10(power[peak])),
                "expected_db": 20.0 + peak_gain_db(CONFIG, window) - CONFIG.coherent_gain_db,
            }
        )
    return rows


def check_oracle_pfa(maps_by_window: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    rows_slice = valid_rows(CONFIG)
    for window, maps in maps_by_window.items():
        cells = maps[:, rows_slice, :].ravel()
        for pfa in (1e-3, 1e-4):
            threshold = threshold_known_noise(pfa)
            rows.append(
                {
                    "check": "oracle_pfa",
                    "window": window,
                    "design_pfa": pfa,
                    "measured_pfa": float(np.mean(cells > threshold)),
                    "cells": int(cells.size),
                }
            )
    return rows


def check_cfar_pfa(maps_by_window: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    rows_slice = valid_rows(CONFIG)
    training = training_cells(CONFIG)
    rank = int(round(0.75 * training))
    for window, maps in maps_by_window.items():
        ca_scores = np.concatenate([(m / ca_estimate(m, CONFIG))[rows_slice].ravel() for m in maps])
        os_scores = np.concatenate([(m / os_estimate(m, CONFIG))[rows_slice].ravel() for m in maps])
        for pfa in (1e-3, 1e-4):
            for name, scores, alpha in (
                ("ca_cfar", ca_scores, ca_alpha_theory(training, pfa)),
                ("os_cfar", os_scores, os_alpha_theory(training, rank, pfa)),
            ):
                rows.append(
                    {
                        "check": "cfar_theory_alpha",
                        "detector": name,
                        "window": window,
                        "design_pfa": pfa,
                        "alpha": alpha,
                        "measured_pfa": float(np.mean(scores > alpha)),
                        "cells": int(scores.size),
                    }
                )
    return rows


def pd_curve(
    swerling: int,
    window: str,
    on_bin: bool,
    snr_grid: np.ndarray,
    frames: int,
    stream: int,
    pfa: float,
) -> np.ndarray:
    threshold = threshold_known_noise(pfa)
    detections = []
    index = 0
    for snr_db in snr_grid:
        hits = 0
        for _ in range(frames):
            rng = frame_rng(ROOT_SEED, stream, index)
            index += 1
            row = rng.uniform(20, 108)
            column = rng.uniform(-32, 32)
            if on_bin:
                row, column = float(round(row)), float(round(column))
            power_db = snr_db
            if swerling == 1:
                power_db = 10 * np.log10(rng.exponential(10 ** (snr_db / 10)))
            target = Target(
                row, column, per_sample_snr_db(power_db, CONFIG), float(rng.uniform(0, 2 * np.pi))
            )
            scene = Scene((target,), white_profile(CONFIG))
            power = simulate_map(scene, CONFIG, rng, window)
            cell = target.nearest_cell(CONFIG)
            hits += int(power[cell] > threshold)
        detections.append(hits / frames)
    return np.array(detections)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=2000)
    parser.add_argument("--pd-frames", type=int, default=400)
    arguments = parser.parse_args()
    output = PROJECT_ROOT / "results" / "baseline"
    output.mkdir(parents=True, exist_ok=True)
    pfa = 1e-4
    rows = check_noise_normalization() + check_target_peak()
    maps_by_window = {
        w: noise_frames(arguments.frames, w, 20 + i) for i, w in enumerate(("rect", "hann"))
    }
    rows += check_oracle_pfa(maps_by_window) + check_cfar_pfa(maps_by_window)
    snr_grid = np.arange(4.0, 24.5, 1.0)
    curves = {
        "swerling0_rect_on_bin": pd_curve(0, "rect", True, snr_grid, arguments.pd_frames, 40, pfa),
        "swerling1_rect_on_bin": pd_curve(1, "rect", True, snr_grid, arguments.pd_frames, 41, pfa),
        "swerling0_hann_off_bin": pd_curve(
            0, "hann", False, snr_grid, arguments.pd_frames, 42, pfa
        ),
        "swerling1_hann_off_bin": pd_curve(
            1, "hann", False, snr_grid, arguments.pd_frames, 43, pfa
        ),
    }
    theory0 = pd_swerling0(10 ** (snr_grid / 10), pfa)
    theory1 = pd_swerling1(10 ** (snr_grid / 10), pfa)
    curve_rows = []
    for i, snr in enumerate(snr_grid):
        row = {
            "integrated_snr_db": float(snr),
            "theory_swerling0": float(theory0[i]),
            "theory_swerling1": float(theory1[i]),
        }
        row.update({name: float(values[i]) for name, values in curves.items()})
        curve_rows.append(row)
    write_csv(curve_rows, output / "pd_vs_theory.csv")
    for name, values in curves.items():
        rows.append(
            {
                "check": "required_snr_pd_0.9",
                "curve": name,
                "measured_db": required_snr_db(snr_grid, values),
                "theory_db": required_snr_db(snr_grid, theory0 if "swerling0" in name else theory1),
            }
        )
    write_csv(rows, output / "checks.csv")
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(snr_grid, theory0, "k-", label="theory Swerling 0")
    axis.plot(snr_grid, theory1, "k--", label="theory Swerling 1")
    for name, values in curves.items():
        axis.plot(
            snr_grid,
            values,
            "o" if "rect" in name else "s",
            markersize=4,
            label=name.replace("_", " "),
        )
    axis.set_xlabel("Integrated SNR (dB, on-bin rectangular reference)")
    axis.set_ylabel("Probability of detection")
    axis.set_title("Known-noise detector, Pfa = 1e-4")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "pd_vs_theory.png", dpi=150)
    plt.close(figure)
    write_json(
        {
            "radar": {
                "carrier_hz": CONFIG.carrier_hz,
                "bandwidth_hz": CONFIG.bandwidth_hz,
                "samples_per_chirp": CONFIG.samples_per_chirp,
                "chirps": CONFIG.chirps,
                "chirp_interval_s": CONFIG.chirp_interval_s,
                "range_resolution_m": CONFIG.range_resolution_m,
                "max_range_m": CONFIG.max_range_m,
                "velocity_resolution_mps": CONFIG.velocity_resolution_mps,
                "max_velocity_mps": CONFIG.max_velocity_mps,
                "coherent_gain_db": CONFIG.coherent_gain_db,
                "hann_peak_gain_db": peak_gain_db(CONFIG, "hann"),
                "training_cells": training_cells(CONFIG),
            },
            "root_seed": ROOT_SEED,
            "noise_frames": arguments.frames,
            "pd_frames_per_point": arguments.pd_frames,
            "versions": package_versions(),
        },
        output / "metadata.json",
    )
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
