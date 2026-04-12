from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SpectralDenoiserMixin:
    n_fft: int
    hop_length: int
    win_length: int

    def _window(self, device: torch.device) -> torch.Tensor:
        return torch.hann_window(self.win_length, device=device)

    def stft(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._window(waveform.device),
            return_complex=True,
        )

    def istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._window(spec.device),
            length=length,
        )


class TinyMaskNet(SpectralDenoiserMixin, nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        hidden_channels: int = 16,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.hidden_channels = hidden_channels

        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, noisy_waveform: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        spec = self.stft(noisy_waveform)
        magnitude = spec.abs()
        log_magnitude = torch.log1p(magnitude).unsqueeze(1)
        mask = self.net(log_magnitude).squeeze(1)
        enhanced_spec = spec * mask
        enhanced_waveform = self.istft(enhanced_spec, length=noisy_waveform.shape[-1])
        return enhanced_waveform, {
            "mask": mask,
            "noisy_spec": spec,
            "enhanced_spec": enhanced_spec,
        }


class FullSubNetLite(SpectralDenoiserMixin, nn.Module):
    """Lightweight FullSubNet-like denoiser for MVP experiments.

    The model combines a full-band temporal branch with a sub-band temporal branch.
    It is intentionally smaller than research FullSubNet/FullSubNet+ variants so it can
    train in short Colab runs.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        full_hidden_size: int = 48,
        sub_hidden_size: int = 24,
        subband_context: int = 2,
        full_num_layers: int = 1,
        sub_num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.full_hidden_size = full_hidden_size
        self.sub_hidden_size = sub_hidden_size
        self.subband_context = subband_context
        self.full_num_layers = full_num_layers
        self.sub_num_layers = sub_num_layers
        self.frequency_bins = n_fft // 2 + 1

        self.full_band = nn.GRU(
            input_size=self.frequency_bins,
            hidden_size=full_hidden_size,
            num_layers=full_num_layers,
            batch_first=True,
        )
        self.full_projection = nn.Sequential(
            nn.Linear(full_hidden_size, self.frequency_bins),
            nn.Tanh(),
        )

        local_size = 2 * subband_context + 1
        self.sub_band = nn.GRU(
            input_size=local_size + 1,
            hidden_size=sub_hidden_size,
            num_layers=sub_num_layers,
            batch_first=True,
        )
        self.mask_projection = nn.Sequential(
            nn.Linear(sub_hidden_size, 1),
            nn.Sigmoid(),
        )

    def _local_frequency_context(self, log_magnitude: torch.Tensor) -> torch.Tensor:
        padded = F.pad(
            log_magnitude.unsqueeze(1),
            pad=(0, 0, self.subband_context, self.subband_context),
            mode="replicate",
        )
        neighborhoods = [
            padded[:, :, offset : offset + self.frequency_bins, :]
            for offset in range(2 * self.subband_context + 1)
        ]
        return torch.cat(neighborhoods, dim=1).permute(0, 2, 3, 1).contiguous()

    def forward(self, noisy_waveform: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        spec = self.stft(noisy_waveform)
        log_magnitude = torch.log1p(spec.abs())

        full_input = log_magnitude.transpose(1, 2)
        full_output, _ = self.full_band(full_input)
        full_cue = self.full_projection(full_output).transpose(1, 2)

        local_context = self._local_frequency_context(log_magnitude)
        full_cue = full_cue.unsqueeze(-1)
        sub_input = torch.cat([local_context, full_cue], dim=-1)

        batch_size, frequency_bins, frames, feature_size = sub_input.shape
        sub_input = sub_input.reshape(batch_size * frequency_bins, frames, feature_size)
        sub_output, _ = self.sub_band(sub_input)
        mask = self.mask_projection(sub_output)
        mask = mask.reshape(batch_size, frequency_bins, frames)

        enhanced_spec = spec * mask
        enhanced_waveform = self.istft(enhanced_spec, length=noisy_waveform.shape[-1])
        return enhanced_waveform, {
            "mask": mask,
            "noisy_spec": spec,
            "enhanced_spec": enhanced_spec,
            "full_cue": full_cue.squeeze(-1),
        }
