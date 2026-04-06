from __future__ import annotations

import json
import random
import shutil
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from .audio import read_audio_mono, write_audio
from .manifests import load_manifest
from .metrics import si_sdr
from .modeling import TinyMaskNet

warnings.filterwarnings(
    "ignore",
    message=r"An output with one or more elements was resized since it had shape \[\]",
    category=UserWarning,
)


@dataclass
class DataConfig:
    train_manifest: Path
    val_manifest: Path
    sample_rate: int
    segment_seconds: float
    batch_size: int
    num_workers: int
    limit_train: int | None
    limit_val: int | None


@dataclass
class ModelConfig:
    n_fft: int
    hop_length: int
    win_length: int
    hidden_channels: int


@dataclass
class TrainConfig:
    epochs: int
    learning_rate: float
    waveform_loss_weight: float
    magnitude_loss_weight: float
    device: str
    output_dir: Path
    checkpoint_mirror_dir: Path | None
    save_every_epoch: bool
    log_interval: int


@dataclass
class ExperimentConfig:
    seed: int
    data: DataConfig
    model: ModelConfig
    training: TrainConfig


class PairedWaveDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        manifest_path: Path,
        sample_rate: int,
        segment_seconds: float,
        random_crop: bool,
        limit: int | None = None,
    ) -> None:
        rows = load_manifest(manifest_path)
        if limit is not None:
            rows = rows[:limit]
        self.rows = rows
        self.sample_rate = sample_rate
        self.segment_samples = int(round(segment_seconds * sample_rate))
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.rows)

    def _crop_or_pad(self, audio: np.ndarray) -> np.ndarray:
        if audio.shape[0] >= self.segment_samples:
            if self.random_crop:
                max_start = audio.shape[0] - self.segment_samples
                start = random.randint(0, max_start) if max_start > 0 else 0
            else:
                start = max(0, (audio.shape[0] - self.segment_samples) // 2)
            return audio[start : start + self.segment_samples].astype(np.float32, copy=False)

        padded = np.zeros(self.segment_samples, dtype=np.float32)
        padded[: audio.shape[0]] = audio
        return padded

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        noisy, _ = read_audio_mono(row["noisy_path"], target_sample_rate=self.sample_rate)
        clean, _ = read_audio_mono(row["clean_path"], target_sample_rate=self.sample_rate)
        noisy = self._crop_or_pad(noisy)
        clean = self._crop_or_pad(clean)
        return torch.from_numpy(noisy), torch.from_numpy(clean)


def resolve_experiment_config(path: Path) -> ExperimentConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    base_dir = path.parent
    data = payload["data"]
    model = payload["model"]
    training = payload["training"]
    return ExperimentConfig(
        seed=int(payload.get("seed", 42)),
        data=DataConfig(
            train_manifest=(base_dir / data["train_manifest"]).resolve(),
            val_manifest=(base_dir / data["val_manifest"]).resolve(),
            sample_rate=int(data["sample_rate"]),
            segment_seconds=float(data["segment_seconds"]),
            batch_size=int(data["batch_size"]),
            num_workers=int(data["num_workers"]),
            limit_train=None if data.get("limit_train") is None else int(data["limit_train"]),
            limit_val=None if data.get("limit_val") is None else int(data["limit_val"]),
        ),
        model=ModelConfig(
            n_fft=int(model["n_fft"]),
            hop_length=int(model["hop_length"]),
            win_length=int(model["win_length"]),
            hidden_channels=int(model["hidden_channels"]),
        ),
        training=TrainConfig(
            epochs=int(training["epochs"]),
            learning_rate=float(training["learning_rate"]),
            waveform_loss_weight=float(training["waveform_loss_weight"]),
            magnitude_loss_weight=float(training["magnitude_loss_weight"]),
            device=str(training["device"]),
            output_dir=(base_dir / training["output_dir"]).resolve(),
            checkpoint_mirror_dir=(
                (base_dir / training["checkpoint_mirror_dir"]).resolve()
                if training.get("checkpoint_mirror_dir")
                else None
            ),
            save_every_epoch=bool(training.get("save_every_epoch", True)),
            log_interval=int(training.get("log_interval", 10)),
        ),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def create_dataloaders(config: ExperimentConfig) -> tuple[DataLoader, DataLoader]:
    train_dataset = PairedWaveDataset(
        manifest_path=config.data.train_manifest,
        sample_rate=config.data.sample_rate,
        segment_seconds=config.data.segment_seconds,
        random_crop=True,
        limit=config.data.limit_train,
    )
    val_dataset = PairedWaveDataset(
        manifest_path=config.data.val_manifest,
        sample_rate=config.data.sample_rate,
        segment_seconds=config.data.segment_seconds,
        random_crop=False,
        limit=config.data.limit_val,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        drop_last=False,
    )
    return train_loader, val_loader


def build_model(config: ExperimentConfig) -> TinyMaskNet:
    return TinyMaskNet(
        n_fft=config.model.n_fft,
        hop_length=config.model.hop_length,
        win_length=config.model.win_length,
        hidden_channels=config.model.hidden_channels,
    )


def magnitude_l1(model: TinyMaskNet, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    estimate_mag = model.stft(estimate).abs()
    target_mag = model.stft(target).abs()
    return torch.mean(torch.abs(torch.log1p(estimate_mag) - torch.log1p(target_mag)))


def batch_si_sdr(estimate: torch.Tensor, target: torch.Tensor) -> float:
    values = []
    estimate_np = estimate.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    for idx in range(estimate_np.shape[0]):
        values.append(si_sdr(target_np[idx], estimate_np[idx]))
    return float(np.mean(values))


def compute_loss(
    model: TinyMaskNet,
    estimate: torch.Tensor,
    target: torch.Tensor,
    config: ExperimentConfig,
) -> torch.Tensor:
    waveform_loss = torch.mean(torch.abs(estimate - target))
    magnitude_loss = magnitude_l1(model, estimate, target)
    return (
        config.training.waveform_loss_weight * waveform_loss
        + config.training.magnitude_loss_weight * magnitude_loss
    )


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_resolved_config(config: ExperimentConfig, output_dir: Path) -> None:
    payload = {
        "seed": config.seed,
        "data": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config.data).items()
        },
        "model": asdict(config.model),
        "training": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config.training).items()
        },
    }
    save_json(output_dir / "resolved_config.json", payload)


def mirror_file(source: Path, mirror_dir: Path | None) -> None:
    if mirror_dir is None:
        return
    mirror_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, mirror_dir / source.name)


def save_checkpoint(
    model: TinyMaskNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    output_dir: Path,
    mirror_dir: Path | None,
    is_best: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "model_config": {
            "n_fft": model.n_fft,
            "hop_length": model.hop_length,
            "win_length": model.win_length,
            "hidden_channels": model.hidden_channels,
        },
    }
    epoch_path = output_dir / f"epoch_{epoch:03d}.pt"
    torch.save(checkpoint, epoch_path)
    torch.save(checkpoint, output_dir / "last.pt")
    mirror_file(epoch_path, mirror_dir)
    mirror_file(output_dir / "last.pt", mirror_dir)
    if is_best:
        torch.save(checkpoint, output_dir / "best.pt")
        mirror_file(output_dir / "best.pt", mirror_dir)


def train_from_config(config_path: Path) -> dict:
    config = resolve_experiment_config(config_path)
    set_seed(config.seed)
    device = choose_device(config.training.device)
    output_dir = config.training.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_resolved_config(config, output_dir)
    mirror_file(output_dir / "resolved_config.json", config.training.checkpoint_mirror_dir)

    train_loader, val_loader = create_dataloaders(config)
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    history: list[dict] = []
    best_val_loss = float("inf")

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        train_losses: list[float] = []
        train_sisdr: list[float] = []
        for _step, (noisy, clean) in enumerate(train_loader, start=1):
            noisy = noisy.to(device)
            clean = clean.to(device)
            optimizer.zero_grad(set_to_none=True)
            estimate, _ = model(noisy)
            loss = compute_loss(model, estimate, clean, config)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.detach().cpu()))
            train_sisdr.append(batch_si_sdr(estimate, clean))

        model.eval()
        val_losses: list[float] = []
        val_sisdr: list[float] = []
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy = noisy.to(device)
                clean = clean.to(device)
                estimate, _ = model(noisy)
                loss = compute_loss(model, estimate, clean, config)
                val_losses.append(float(loss.detach().cpu()))
                val_sisdr.append(batch_si_sdr(estimate, clean))

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
            "train_si_sdr": float(np.mean(train_sisdr)) if train_sisdr else float("nan"),
            "val_loss": float(np.mean(val_losses)) if val_losses else float("nan"),
            "val_si_sdr": float(np.mean(val_sisdr)) if val_sisdr else float("nan"),
            "device": str(device),
        }
        history.append(epoch_metrics)
        save_json(output_dir / "history.json", {"history": history})
        mirror_file(output_dir / "history.json", config.training.checkpoint_mirror_dir)

        is_best = epoch_metrics["val_loss"] < best_val_loss
        if is_best:
            best_val_loss = epoch_metrics["val_loss"]

        if config.training.save_every_epoch:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=epoch_metrics,
                output_dir=output_dir,
                mirror_dir=config.training.checkpoint_mirror_dir,
                is_best=is_best,
            )

    return {
        "epochs": len(history),
        "best_val_loss": best_val_loss,
        "best_val_si_sdr": max((item["val_si_sdr"] for item in history), default=float("nan")),
        "device": str(device),
        "output_dir": str(output_dir),
    }


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: str = "auto",
) -> tuple[TinyMaskNet, torch.device]:
    torch_device = choose_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=torch_device)
    model_config = checkpoint["model_config"]
    model = TinyMaskNet(
        n_fft=int(model_config["n_fft"]),
        hop_length=int(model_config["hop_length"]),
        win_length=int(model_config["win_length"]),
        hidden_channels=int(model_config["hidden_channels"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch_device)
    model.eval()
    return model, torch_device


def enhance_manifest(
    checkpoint_path: Path,
    rendered_manifest_path: Path,
    output_dir: Path,
    device: str = "auto",
    limit: int | None = None,
) -> Path:
    model, torch_device = load_model_from_checkpoint(checkpoint_path, device=device)
    rows = load_manifest(rendered_manifest_path)
    if limit is not None:
        rows = rows[:limit]

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []
    for row in rows:
        noisy_path = Path(row["noisy_path"])
        clean_path = row.get("clean_path")
        noisy, sample_rate = read_audio_mono(noisy_path)
        if sample_rate <= 0:
            raise ValueError(f"Некорректный sample rate в {noisy_path}")
        waveform = torch.from_numpy(noisy).unsqueeze(0).to(torch_device)
        with torch.no_grad():
            enhanced, _ = model(waveform)
        enhanced_np = enhanced.squeeze(0).detach().cpu().numpy()
        enhanced_path = output_dir / f"{row['id']}.wav"
        write_audio(enhanced_path, enhanced_np, sample_rate)
        manifest_rows.append(
            {
                "id": row["id"],
                "noisy_path": str(noisy_path),
                "clean_path": clean_path,
                "enhanced_path": str(enhanced_path.resolve()),
            }
        )

    manifest_path = output_dir / "enhanced_manifest.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in manifest_rows) + "\n",
        encoding="utf-8",
    )
    return manifest_path
