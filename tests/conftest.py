import importlib.util

import numpy as np
import pytest

from neural_radar.config import DEFAULT_CONFIG
from neural_radar.scene import Scene, Target, white_profile

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

requires_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")


def single_target_scene(range_bin: float, doppler_bin: float, snr_in_db: float = 0.0) -> Scene:
    return Scene((Target(range_bin, doppler_bin, snr_in_db),), white_profile(DEFAULT_CONFIG))


@pytest.fixture
def config():
    return DEFAULT_CONFIG


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
