from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from neural_radar.artifacts import load_protocol
from neural_radar.cfar import CLASSICAL_DETECTORS
from neural_radar.config import DEFAULT_CONFIG, PROJECT_ROOT, RadarConfig
from neural_radar.evaluation import Scorer
from neural_radar.simulation import Frame, generate_frames

DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "detection" / "protocol.json"


@dataclass(frozen=True)
class DetectionProtocol:
    values: dict
    config: RadarConfig = DEFAULT_CONFIG

    @classmethod
    def load(cls, path: Path = DEFAULT_PROTOCOL) -> "DetectionProtocol":
        protocol = cls(load_protocol(path))
        protocol.validate()
        return protocol

    def validate(self) -> None:
        seeds = [
            self.values["calibration"]["seed"],
            self.values["test"]["seed"],
            self.values["training"]["seed"],
            self.values["diagnostics"]["seed"],
        ]
        if len(set(seeds)) != len(seeds):
            raise ValueError(
                "Calibration, test, training and diagnostic frames must use different root seeds"
            )
        for kind in self.values["training"]["mixture"]:
            if not self.values["scenarios"][kind]["in_training"]:
                raise ValueError(
                    f"Scenario {kind} is marked out-of-distribution but used in training"
                )

    @property
    def false_alarm_rate(self) -> float:
        return float(self.values["false_alarm_rate"])

    @property
    def scenarios(self) -> list[str]:
        return list(self.values["test"]["scenarios"])

    def path(self, key: str) -> Path:
        return PROJECT_ROOT / self.values["paths"][key]

    def calibration_frames(self) -> Iterator[Frame]:
        calibration = self.values["calibration"]
        settings = dict(self.values["scenarios"]["homogeneous"])
        settings["targets"] = [0, 0]
        return generate_frames(
            "homogeneous",
            [float("nan")],
            calibration["frames"],
            settings,
            self.config,
            calibration["seed"],
            calibration["stream"],
            self.values["window"],
        )

    def test_frames(self, scenario: str) -> Iterator[Frame]:
        test = self.values["test"]
        return generate_frames(
            scenario,
            [float(value) for value in test["snr_db"]],
            test["frames_per_snr"],
            self.values["scenarios"][scenario],
            self.config,
            test["seed"],
            1 + self.scenarios.index(scenario),
            self.values["window"],
        )


def classical_scorers(config: RadarConfig) -> dict[str, Scorer]:
    return {
        detector.name: (lambda power, detector=detector: detector.score(power, config))
        for detector in CLASSICAL_DETECTORS
    }
