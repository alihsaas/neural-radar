import csv
import json
import platform
from importlib import metadata
from pathlib import Path

PACKAGES = ("numpy", "scipy", "matplotlib", "torch")


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def load_protocol(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="ascii"))


def write_json(values: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, sort_keys=False) + "\n", encoding="ascii")


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(value) for key, value in row.items()})


def format_value(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="ascii") as handle:
        return list(csv.DictReader(handle))
