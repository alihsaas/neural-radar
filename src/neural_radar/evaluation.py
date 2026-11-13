from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

from neural_radar.config import RadarConfig
from neural_radar.metrics import (
    detection_rate,
    noise_scores,
    required_snr_db,
    target_scores,
    threshold_for_pfa,
    wilson_interval,
)
from neural_radar.simulation import Frame

Scorer = Callable[[np.ndarray], np.ndarray]
TAIL_SIZE = 250_000


@dataclass
class ScoreSummary:
    target: list[np.ndarray] = field(default_factory=list)
    target_snr_db: list[np.ndarray] = field(default_factory=list)
    noise_tail: np.ndarray = field(default_factory=lambda: np.zeros(0))
    noise_cells: int = 0

    def add(self, score: np.ndarray, cells: np.ndarray, snr_db: float, config: RadarConfig) -> None:
        self.target.append(target_scores(score, cells, config))
        self.target_snr_db.append(np.full(len(cells), snr_db))
        noise = noise_scores(score, cells, config)
        self.noise_cells += noise.size
        merged = np.concatenate([self.noise_tail, noise])
        if merged.size > 2 * TAIL_SIZE:
            merged = np.partition(merged, merged.size - TAIL_SIZE)[-TAIL_SIZE:]
        self.noise_tail = merged

    def targets(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.target:
            return np.zeros(0), np.zeros(0)
        return np.concatenate(self.target), np.concatenate(self.target_snr_db)

    def false_alarms(self, threshold: float) -> int:
        return int(np.sum(self.noise_tail > threshold))

    def saturated(self, threshold: float) -> bool:
        count = self.false_alarms(threshold)
        return count >= self.noise_tail.size and self.noise_cells > self.noise_tail.size

    def false_alarm_rate(self, threshold: float) -> float:
        return self.false_alarms(threshold) / max(self.noise_cells, 1)

    def threshold_for(self, false_alarm_rate: float) -> float:
        rank = int(np.floor(false_alarm_rate * self.noise_cells))
        if rank < 1 or rank >= self.noise_tail.size:
            return float("nan")
        return float(np.sort(self.noise_tail)[::-1][rank])


def save_summaries(path, summaries: dict[str, dict[str, ScoreSummary]]) -> None:
    arrays = {}
    for scenario, by_name in summaries.items():
        for name, summary in by_name.items():
            target, snr = summary.targets()
            prefix = f"{scenario}__{name}__"
            arrays[prefix + "target"] = target
            arrays[prefix + "snr"] = snr
            arrays[prefix + "tail"] = summary.noise_tail
            arrays[prefix + "cells"] = np.array([summary.noise_cells])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_summaries(path) -> dict[str, dict[str, ScoreSummary]]:
    summaries: dict[str, dict[str, ScoreSummary]] = {}
    with np.load(path) as stored:
        for key in stored.files:
            scenario, name, part = key.split("__")
            summary = summaries.setdefault(scenario, {}).setdefault(name, ScoreSummary())
            if part == "target":
                summary.target = [stored[key]]
            elif part == "snr":
                summary.target_snr_db = [stored[key]]
            elif part == "tail":
                summary.noise_tail = stored[key]
            else:
                summary.noise_cells = int(stored[key][0])
    return summaries


def score_frames(
    frames: Iterable[Frame], scorers: dict[str, Scorer], config: RadarConfig
) -> dict[str, ScoreSummary]:
    summaries = {name: ScoreSummary() for name in scorers}
    for frame in frames:
        cells = frame.scene.target_cells(config)
        for name, scorer in scorers.items():
            summaries[name].add(scorer(frame.power), cells, frame.scene.integrated_snr_db, config)
    return summaries


def calibrate(
    frames: Iterable[Frame],
    scorers: dict[str, Scorer],
    config: RadarConfig,
    false_alarm_rate: float,
) -> dict[str, float]:
    pools: dict[str, list[np.ndarray]] = {name: [] for name in scorers}
    for frame in frames:
        cells = frame.scene.target_cells(config)
        for name, scorer in scorers.items():
            pools[name].append(noise_scores(scorer(frame.power), cells, config))
    return {
        name: threshold_for_pfa(np.concatenate(values), false_alarm_rate)
        for name, values in pools.items()
    }


def summary_rows(
    scenario: str,
    summaries: dict[str, ScoreSummary],
    thresholds: dict[str, float],
    roc_rates: list[float],
    extra: dict[str, dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    curve_rows = []
    overview_rows = []
    for name, summary in summaries.items():
        target, snr = summary.targets()
        threshold = thresholds[name]
        detection_by_snr = []
        levels = sorted(set(snr.tolist()))
        for level in levels:
            chosen = target[snr == level]
            hits = int(np.sum(chosen > threshold))
            low, high = wilson_interval(hits, chosen.size)
            detection = hits / chosen.size if chosen.size else float("nan")
            detection_by_snr.append(detection)
            curve_rows.append(
                {
                    "scenario": scenario,
                    "detector": name,
                    "snr_db": level,
                    "targets": int(chosen.size),
                    "detected": hits,
                    "pd": detection,
                    "pd_ci_low": low,
                    "pd_ci_high": high,
                }
            )
        row = {
            "scenario": scenario,
            "detector": name,
            "threshold": threshold,
            "targets": int(target.size),
            "noise_cells": summary.noise_cells,
            "false_alarms": summary.false_alarms(threshold),
            "measured_pfa": summary.false_alarm_rate(threshold),
            "pfa_is_lower_bound": summary.saturated(threshold),
            "pd_all_snr": detection_rate(target, threshold),
            "snr_for_pd_0.5_db": required_snr_db(np.array(levels), np.array(detection_by_snr), 0.5),
            "snr_for_pd_0.8_db": required_snr_db(np.array(levels), np.array(detection_by_snr), 0.8),
        }
        for rate in roc_rates:
            local = summary.threshold_for(rate)
            row[f"pd_at_scenario_pfa_{rate:g}"] = (
                detection_rate(target, local) if np.isfinite(local) else float("nan")
            )
        if extra and name in extra:
            row.update(extra[name])
        overview_rows.append(row)
    return curve_rows, overview_rows
