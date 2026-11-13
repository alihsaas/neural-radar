import argparse
import time
from pathlib import Path

import torch

from neural_radar.artifacts import package_versions, write_csv, write_json
from neural_radar.experiment import DEFAULT_PROTOCOL, DetectionProtocol
from neural_radar.neural import RunSettings, checkpoint_payload, parameter_count, train_model
from neural_radar.neural_data import training_datasets


def run_settings(
    protocol: DetectionProtocol, run: dict, seed: int, epochs: int | None
) -> RunSettings:
    neural = protocol.values["neural"]
    return RunSettings(
        name=run["name"],
        architecture=run["architecture"],
        seed=seed,
        epochs=epochs or neural["epochs"],
        batch_size=neural["batch_size"],
        learning_rate=neural["learning_rate"],
        weight_decay=neural["weight_decay"],
        positive_weight=neural["positive_weight"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--run", nargs="*", default=None)
    parser.add_argument("--seed", type=int, nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    arguments = parser.parse_args()
    protocol = DetectionProtocol.load(arguments.protocol)
    output = arguments.output or protocol.path("checkpoints")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.perf_counter()
    train, validation = training_datasets(protocol, protocol.path("datasets"))
    print(
        f"device={device} train={len(train)} validation={len(validation)} "
        f"train_targets={int(train.labels.sum())} data_seconds={time.perf_counter() - start:.0f}",
        flush=True,
    )
    runs = [
        r
        for r in protocol.values["neural"]["runs"]
        if arguments.run is None or r["name"] in arguments.run
    ]
    seeds = arguments.seed or protocol.values["neural"]["seeds"]
    for run in runs:
        for seed in seeds:
            path = output / f"{run['name']}_seed{seed}.pt"
            if arguments.skip_existing and path.exists():
                print(f"skip {path.name}")
                continue
            settings = run_settings(protocol, run, seed, arguments.epochs)
            start = time.perf_counter()
            model, history, best_epoch = train_model(settings, train, validation, device)
            seconds = time.perf_counter() - start
            torch.save(checkpoint_payload(model, settings, history, best_epoch), path)
            write_csv(history, output / f"{run['name']}_seed{seed}_history.csv")
            print(
                f"{run['name']} seed={seed} params={parameter_count(model)} best_epoch={best_epoch} "
                f"val_loss={history[best_epoch - 1]['validation_loss']:.5f} seconds={seconds:.0f}",
                flush=True,
            )
    write_json(
        {
            "protocol": protocol.values,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "versions": package_versions(),
        },
        output / "training_metadata.json",
    )


if __name__ == "__main__":
    main()
