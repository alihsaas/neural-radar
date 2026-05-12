# NeuralRadar

I use this project to study target detection on simulated radar frames.

## Setup

I use Python 3.12 and uv. Install the locked dependencies with `uv sync --locked`.

## Simulator

I generate simulated FMCW returns and convert them to range-Doppler maps.

## Classical detectors

I implemented CA-, GO-, and OS-CFAR and a detector that uses the known noise level.

## Tests

Run `uv run pytest -q --basetemp=.pytest-tmp`.

## Benchmark

Run `uv run python scripts/verify_baseline.py` to check the simulator against theory and `uv run python scripts/run_classical_benchmark.py` to compare detectors.
