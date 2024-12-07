import torch

from src.metrics.base_metric import BaseMetric
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility


class STOI(BaseMetric):
    def __init__(self, device="auto", fs=16000, extended=False, *args, **kwargs):
        """
        STOI (Short-Time Objective Intelligibility) metric.

        Args:
            fs (int): Sample rate of the audio signals.
            extended (bool): If True, use the extended STOI variant.
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = ShortTimeObjectiveIntelligibility(fs, extended=extended).to(device)

    def __call__(
            self,
            audio: torch.Tensor,  # Ground truth audio [B, 1, T]
            pred_audio: torch.Tensor,  # Predicted audio [B, 1, T]
            **kwargs
    ):
        """
        Compute STOI between predicted and reference audio.

        Args:
            audio_ref (Tensor): reference audio [B, 1, T]
            pred_audio (Tensor): predicted audio [B, 1, T]

        Returns:
            (float): STOI score averaged over the batch.
        """
        # make audio_ref and pred_audio same T
        if audio.shape[-1] != pred_audio.shape[-1]:
            min_len = min(audio.shape[-1], pred_audio.shape[-1])
            audio = audio[..., :min_len]
            pred_audio = pred_audio[..., :min_len]
        return self.metric(pred_audio, audio)
