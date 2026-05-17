import numpy as np
import pytest
from conftest import requires_torch, single_target_scene

from neural_radar.neural_data import label_maps, network_input
from neural_radar.simulation import Frame


def test_label_maps_mark_target_and_ignore_its_surroundings(config):
    scene = single_target_scene(60.0, 0.0)
    frame = Frame(scene, np.ones((config.samples_per_chirp, config.chirps)))
    label, weight = label_maps(frame, config)
    assert label.sum() == 1 and label[60, 32] == 1
    assert weight[60, 32] == 1
    assert weight[61, 33] == 0 and weight[62, 34] == 0
    assert weight[63, 32] == 1
    assert weight[5].sum() == 0


def test_network_input_is_median_centred(config, rng):
    power = rng.exponential(2.0, size=(config.samples_per_chirp, config.chirps))
    values = network_input(power)
    assert values.dtype == np.float32
    assert np.median(values) == pytest.approx(0.0, abs=1e-3)


@requires_torch
@pytest.mark.parametrize("architecture", ["local", "unet"])
def test_models_keep_map_shape(architecture):
    import torch

    from neural_radar.neural import build_model

    torch.manual_seed(0)
    model = build_model(architecture).eval()
    with torch.no_grad():
        output = model(torch.randn(2, 128, 64))
    assert output.shape == (2, 128, 64)


@requires_torch
def test_local_model_is_doppler_shift_equivariant():
    import torch

    from neural_radar.neural import build_model

    torch.manual_seed(0)
    model = build_model("local").eval()
    values = torch.randn(2, 128, 64)
    with torch.no_grad():
        output = model(values)
        shifted = model(torch.roll(values, shifts=8, dims=-1))
    assert torch.allclose(torch.roll(output, shifts=8, dims=-1), shifted, atol=1e-4)


@requires_torch
def test_weighted_loss_ignores_zero_weight_cells():
    import torch

    from neural_radar.neural import weighted_loss

    logits = torch.zeros(1, 4, 4)
    labels = torch.zeros(1, 4, 4)
    weights = torch.ones(1, 4, 4)
    base = float(weighted_loss(logits, labels, weights, 30.0))
    logits[0, 0, 0] = 50.0
    weights[0, 0, 0] = 0.0
    assert float(weighted_loss(logits, labels, weights, 30.0)) == pytest.approx(base)
