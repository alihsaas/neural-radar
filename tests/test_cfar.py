import numpy as np
import pytest

from neural_radar.cfar import (
    box_sum,
    ca_alpha_theory,
    ca_estimate,
    ca_pfa_theory,
    go_estimate,
    os_alpha_theory,
    os_estimate,
    os_pfa_theory,
    training_cells,
    training_mask,
)


def test_training_ring_size(config):
    mask = training_mask(config)
    assert mask.shape == (21, 13)
    assert training_cells(config) == 21 * 13 - 5 * 5 == 248
    assert not mask[10, 6]


@pytest.mark.parametrize("estimate", [ca_estimate, go_estimate, os_estimate])
def test_constant_map_gives_constant_estimate(config, estimate):
    power = np.full((config.samples_per_chirp, config.chirps), 3.5)
    assert np.allclose(estimate(power, config), 3.5)


def test_cell_under_test_does_not_enter_its_own_estimate(config):
    power = np.ones((config.samples_per_chirp, config.chirps))
    power[60, 30] = 1e6
    assert ca_estimate(power, config)[60, 30] == pytest.approx(1.0)
    assert os_estimate(power, config)[60, 30] == pytest.approx(1.0)


def test_greatest_of_reacts_to_one_sided_step(config):
    power = np.ones((config.samples_per_chirp, config.chirps))
    power[:62] = 30.0
    ca = ca_estimate(power, config)[62, 10]
    go = go_estimate(power, config)[62, 10]
    assert go == pytest.approx(30.0)
    assert ca < go


def test_ca_theory_round_trip():
    alpha = ca_alpha_theory(248, 1e-4)
    assert ca_pfa_theory(alpha, 248) == pytest.approx(1e-4, rel=1e-9)


def test_os_theory_round_trip():
    alpha = os_alpha_theory(248, 186, 1e-4)
    assert os_pfa_theory(alpha, 248, 186) == pytest.approx(1e-4, rel=1e-6)


def test_ca_cfar_matches_theory_on_white_exponential_noise(config, rng):
    alpha = ca_alpha_theory(training_cells(config), 1e-2)
    exceedances = 0
    cells = 0
    for _ in range(40):
        power = rng.exponential(1.0, size=(config.samples_per_chirp, config.chirps))
        exceedances += int(np.sum(power > alpha * ca_estimate(power, config)))
        cells += power.size
    assert exceedances / cells == pytest.approx(1e-2, rel=0.08)


def test_box_sum_collects_energy_split_over_neighbours(config):
    power = np.zeros((config.samples_per_chirp, config.chirps))
    power[40, 0] = 1.0
    power[41, 63] = 2.0
    assert box_sum(power, 3)[40, 63] == pytest.approx(3.0)
