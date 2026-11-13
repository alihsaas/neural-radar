import numpy as np
import pytest
from conftest import single_target_scene

from neural_radar.scene import Scene, edge_profile, white_profile
from neural_radar.signal import peak_gain_db, range_doppler_power, simulate_map, target_signal


@pytest.mark.parametrize("window", ["rect", "hann"])
def test_noise_power_per_cell_is_one(config, rng, window):
    scene = Scene((), white_profile(config))
    maps = np.stack([simulate_map(scene, config, rng, window) for _ in range(40)])
    assert maps.mean() == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("window", ["rect", "hann"])
def test_on_bin_target_peaks_on_its_cell_with_expected_gain(config, window):
    scene = single_target_scene(40.0, 7.0)
    power = range_doppler_power(target_signal(scene, config), config, window)
    row, column = np.unravel_index(np.argmax(power), power.shape)
    assert [int(row), int(column)] == scene.target_cells(config)[0].tolist() == [40, 39]
    assert 10 * np.log10(power[row, column]) == pytest.approx(
        peak_gain_db(config, window), abs=1e-6
    )


def test_rect_peak_gain_equals_coherent_gain(config):
    assert peak_gain_db(config, "rect") == pytest.approx(config.coherent_gain_db)


def test_negative_doppler_maps_below_centre_column(config):
    scene = single_target_scene(50.0, -10.0)
    power = range_doppler_power(target_signal(scene, config), config, "rect")
    assert np.unravel_index(np.argmax(power), power.shape) == (50, 22)


def test_edge_profile_raises_noise_only_before_edge(config, rng):
    scene = Scene((), edge_profile(config, 60, 15.0))
    maps = np.stack([simulate_map(scene, config, rng) for _ in range(30)])
    near = maps[:, 15:55].mean()
    far = maps[:, 65:118].mean()
    assert 10 * np.log10(near / far) == pytest.approx(10 * np.log10(1 + 10**1.5), abs=0.5)
