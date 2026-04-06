from __future__ import annotations

import torch
from torch import nn


class TinyMaskNet(nn.Module):
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
