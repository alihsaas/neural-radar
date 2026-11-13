from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEED_OF_LIGHT = 299_792_458.0


@dataclass(frozen=True)
class RadarConfig:
    carrier_hz: float = 77e9
    bandwidth_hz: float = 150e6
    samples_per_chirp: int = 128
    chirps: int = 64
    chirp_interval_s: float = 40e-6
    guard_range: int = 2
    guard_doppler: int = 2
    train_range: int = 8
    train_doppler: int = 4

    @property
    def wavelength_m(self) -> float:
        return SPEED_OF_LIGHT / self.carrier_hz

    @property
    def range_resolution_m(self) -> float:
        return SPEED_OF_LIGHT / (2.0 * self.bandwidth_hz)

    @property
    def max_range_m(self) -> float:
        return self.samples_per_chirp * self.range_resolution_m

    @property
    def velocity_resolution_mps(self) -> float:
        return self.wavelength_m / (2.0 * self.chirps * self.chirp_interval_s)

    @property
    def max_velocity_mps(self) -> float:
        return self.wavelength_m / (4.0 * self.chirp_interval_s)

    @property
    def range_margin(self) -> int:
        return self.guard_range + self.train_range

    @property
    def valid_range_bins(self) -> range:
        return range(self.range_margin, self.samples_per_chirp - self.range_margin)

    @property
    def coherent_gain_db(self) -> float:
        from math import log10

        return 10.0 * log10(self.samples_per_chirp * self.chirps)


DEFAULT_CONFIG = RadarConfig()
