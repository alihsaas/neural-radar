from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from neural_radar.neural_data import DetectionDataset, network_input


@dataclass(frozen=True)
class RunSettings:
    name: str
    architecture: str
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    positive_weight: float


class RadarConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = functional.pad(values, (1, 1, 0, 0), mode="circular")
        values = functional.pad(values, (0, 0, 1, 1), mode="replicate")
        return functional.relu(self.norm(self.conv(values)))


def double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        RadarConv(in_channels, out_channels), RadarConv(out_channels, out_channels)
    )


class LocalDetector(nn.Module):
    def __init__(self, width: int = 16, layers: int = 4):
        super().__init__()
        blocks = [RadarConv(1, width)] + [RadarConv(width, width) for _ in range(layers - 1)]
        self.body = nn.Sequential(*blocks)
        self.head = nn.Conv2d(width, 1, kernel_size=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(values.unsqueeze(1))).squeeze(1)


class UNetDetector(nn.Module):
    def __init__(self, width: int = 16):
        super().__init__()
        self.encoder1 = double_conv(1, width)
        self.encoder2 = double_conv(width, 2 * width)
        self.bottleneck = double_conv(2 * width, 4 * width)
        self.decoder2 = double_conv(6 * width, 2 * width)
        self.decoder1 = double_conv(3 * width, width)
        self.head = nn.Conv2d(width, 1, kernel_size=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        level1 = self.encoder1(values.unsqueeze(1))
        level2 = self.encoder2(functional.max_pool2d(level1, 2))
        bottom = self.bottleneck(functional.max_pool2d(level2, 2))
        up2 = functional.interpolate(bottom, size=level2.shape[-2:], mode="nearest")
        decoded2 = self.decoder2(torch.cat([up2, level2], dim=1))
        up1 = functional.interpolate(decoded2, size=level1.shape[-2:], mode="nearest")
        decoded1 = self.decoder1(torch.cat([up1, level1], dim=1))
        return self.head(decoded1).squeeze(1)


def build_model(architecture: str) -> nn.Module:
    if architecture == "unet":
        return UNetDetector()
    if architecture == "local":
        return LocalDetector()
    raise ValueError(f"Unknown architecture: {architecture}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def receptive_field(architecture: str) -> str:
    return {"local": "9 x 9 cells", "unet": "about 44 x 44 cells"}[architecture]


def seed_everything(seed: int, device: torch.device) -> torch.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def weighted_loss(
    logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor, positive_weight: float
) -> torch.Tensor:
    per_cell = functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=torch.tensor(positive_weight, device=logits.device),
        reduction="none",
    )
    return (per_cell * weights).sum() / weights.sum().clamp_min(1.0)


def augment(
    inputs: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor, generator: torch.Generator
):
    shift = int(
        torch.randint(0, inputs.shape[-1], (1,), generator=generator, device=inputs.device).item()
    )
    flip = bool(torch.rand(1, generator=generator, device=inputs.device).item() < 0.5)
    tensors = [torch.roll(t, shifts=shift, dims=-1) for t in (inputs, labels, weights)]
    if flip:
        tensors = [torch.flip(t, dims=(-1,)) for t in tensors]
    return tensors


def dataset_tensors(dataset: DetectionDataset, device: torch.device):
    return (
        torch.from_numpy(dataset.inputs).to(device),
        torch.from_numpy(dataset.labels).to(device),
        torch.from_numpy(dataset.weights).to(device),
    )


@torch.no_grad()
def validation_loss(model: nn.Module, tensors, batch_size: int, positive_weight: float) -> float:
    model.eval()
    inputs, labels, weights = tensors
    total = 0.0
    weight_sum = 0.0
    for start in range(0, inputs.shape[0], batch_size):
        stop = start + batch_size
        device = next(model.parameters()).device
        logits = model(inputs[start:stop].to(device).float())
        batch_weights = weights[start:stop].to(device).float()
        batch_labels = labels[start:stop].to(device).float()
        total += float(weighted_loss(logits, batch_labels, batch_weights, positive_weight)) * float(
            batch_weights.sum()
        )
        weight_sum += float(batch_weights.sum())
    return total / weight_sum


def train_model(
    settings: RunSettings,
    train: DetectionDataset,
    validation: DetectionDataset,
    device: torch.device,
):
    generator = seed_everything(settings.seed, device)
    model = build_model(settings.architecture).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.epochs)
    cpu = torch.device("cpu")
    train_inputs, train_labels, train_weights = dataset_tensors(train, cpu)
    validation_tensors = dataset_tensors(validation, cpu)
    history = []
    best_state = None
    best_epoch = -1
    best_loss = float("inf")
    for epoch in range(1, settings.epochs + 1):
        model.train()
        order = torch.randperm(train_inputs.shape[0], generator=generator, device=device).cpu()
        running = 0.0
        batches = 0
        for start in range(0, order.shape[0], settings.batch_size):
            batch = order[start : start + settings.batch_size]
            inputs, labels, weights = augment(
                train_inputs[batch].to(device).float(),
                train_labels[batch].to(device).float(),
                train_weights[batch].to(device).float(),
                generator,
            )
            loss = weighted_loss(model(inputs), labels, weights, settings.positive_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            batches += 1
        scheduler.step()
        current = validation_loss(
            model, validation_tensors, settings.batch_size, settings.positive_weight
        )
        history.append(
            {"epoch": epoch, "train_loss": running / batches, "validation_loss": current}
        )
        if current < best_loss:
            best_loss = current
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


def checkpoint_payload(
    model: nn.Module, settings: RunSettings, history: list[dict], best_epoch: int
) -> dict:
    return {
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "settings": asdict(settings),
        "history": history,
        "best_epoch": best_epoch,
        "parameters": parameter_count(model),
    }


def load_checkpoint(path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=True)
    settings = RunSettings(**payload["settings"])
    model = build_model(settings.architecture).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, settings, payload


def make_scorer(model: nn.Module, device: torch.device):
    @torch.no_grad()
    def score(power: np.ndarray) -> np.ndarray:
        values = torch.from_numpy(network_input(power)).to(device).unsqueeze(0)
        return model(values)[0].float().cpu().numpy().astype(np.float64)

    return score


@torch.no_grad()
def score_batch(
    model: nn.Module, inputs: np.ndarray, device: torch.device, batch_size: int = 64
) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, inputs.shape[0], batch_size):
        values = (
            torch.from_numpy(np.ascontiguousarray(inputs[start : start + batch_size]))
            .to(device)
            .float()
        )
        outputs.append(model(values).float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)
