import torch
from src.metrics.base_metric import BaseMetric

class LSD(BaseMetric):
    def __init__(
        self,
        device="auto",
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = None,
        eps: float = 1e-12,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.eps = eps

    def __call__(
        self,
        audio: torch.Tensor,
        pred_audio: torch.Tensor,
        **kwargs
    ) -> float:
        audio = audio.to(self.device)
        pred_audio = pred_audio.to(self.device)

        min_len = min(audio.shape[-1], pred_audio.shape[-1])
        audio = audio[..., :min_len]
        pred_audio = pred_audio[..., :min_len]

        spec_ref = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True
        )
        spec_pred = torch.stft(
            pred_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True
        )

        mag_ref = torch.abs(spec_ref)
        mag_pred = torch.abs(spec_pred)

        log_ref = 20.0 * torch.log10(mag_ref.clamp(min=self.eps))
        log_pred = 20.0 * torch.log10(mag_pred.clamp(min=self.eps))

        diff2 = (log_pred - log_ref) ** 2

        lsd_per_frame = torch.sqrt(diff2.mean(dim=-2) + self.eps)
        lsd = lsd_per_frame.mean().item()

        return lsd
