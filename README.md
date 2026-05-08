# NeuralRadar

I use this project to study target detection on simulated radar frames.

## Setup

I use Python 3.12 and uv. Install the locked dependencies with `uv sync --locked`.

## Simulator

I generate simulated FMCW returns and convert them to range-Doppler maps.

## Classical detectors

I implemented CA-, GO-, and OS-CFAR and a detector that uses the known noise level.
