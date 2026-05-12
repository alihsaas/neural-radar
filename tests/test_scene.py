import numpy as np
import pytest

from neural_radar.experiment import DetectionProtocol
from neural_radar.scene import build_scene
from neural_radar.simulation import frame_rng, generate_frames

KINDS = ["homogeneous", "dense", "clutter_edge", "k_clutter", "interference"]


@pytest.fixture(scope="module")
def protocol():
    return DetectionProtocol.load()


def scenes(protocol, kind, stream, count=30):
    settings = protocol.values["scenarios"][kind]
    return [
        build_scene(kind, 15.0, protocol.config, frame_rng(1, stream, i), settings)
        for i in range(count)
    ]


@pytest.mark.parametrize("kind", KINDS)
def test_targets_stay_inside_valid_rows(protocol, kind):
    config = protocol.config
    for scene in scenes(protocol, kind, 2):
        cells = scene.target_cells(config)
        assert np.all(cells[:, 0] >= config.valid_range_bins.start)
        assert np.all(cells[:, 0] < config.valid_range_bins.stop)


def test_clutter_edge_targets_are_beyond_the_edge(protocol):
    for scene in scenes(protocol, "clutter_edge", 3):
        assert all(t.range_bin >= scene.metadata["edge_bin"] + 2 for t in scene.targets)


def test_dense_targets_keep_minimum_separation(protocol):
    chirps = protocol.config.chirps
    for scene in scenes(protocol, "dense", 4):
        for i, a in enumerate(scene.targets):
            for b in scene.targets[i + 1 :]:
                gap = abs(a.doppler_bin - b.doppler_bin) % chirps
                gap = min(gap, chirps - gap)
                assert max(abs(a.range_bin - b.range_bin), gap) >= 3.0


def test_frames_are_reproducible(protocol):
    settings = protocol.values["scenarios"]["homogeneous"]
    first = next(generate_frames("homogeneous", [12.0], 1, settings, protocol.config, 9, 1))
    second = next(generate_frames("homogeneous", [12.0], 1, settings, protocol.config, 9, 1))
    assert np.array_equal(first.power, second.power)


def test_protocol_rejects_shared_seeds(protocol):
    values = dict(protocol.values)
    values["test"] = dict(values["test"], seed=values["training"]["seed"])
    with pytest.raises(ValueError):
        DetectionProtocol(values).validate()


def test_protocol_rejects_out_of_distribution_scenarios_in_training(protocol):
    values = dict(protocol.values)
    values["training"] = dict(values["training"], mixture={"k_clutter": 1.0})
    with pytest.raises(ValueError):
        DetectionProtocol(values).validate()
