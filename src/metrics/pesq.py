import torch

from src.metrics.base_metric import BaseMetric
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality


class PESQ(BaseMetric):
    def __init__(self, device="auto", fs=16000, mode="wb", n_processes=1, *args, **kwargs):
        """
        PESQ (Perceptual Evaluation of Speech Quality) metric.

        Args:
            fs (int): Sample rate of the audio signals (Hz).
            mode (str): "wb" (wide-band) or "nb" (narrow-band).
            n_processes (int): Number of parallel processes to speed up PESQ calculation.
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = PerceptualEvaluationSpeechQuality(fs, mode=mode, n_processes=n_processes).to(device)

    def __call__(
            self,
            audio: torch.Tensor,  # Ground truth audio [B, 1, T]
            pred_audio: torch.Tensor,  # Predicted audio [B, 1, T]
            **kwargs
    ):
        """
        Compute PESQ between predicted and reference audio.

        Args:
            audio_ref (Tensor): reference audio [B, 1, T]
            pred_audio (Tensor): predicted audio [B, 1, T]

        Returns:
            (float): PESQ score averaged over the batch.
        """
        # make audio_ref and pred_audio same T
        if audio.shape[-1] != pred_audio.shape[-1]:
            min_len = min(audio.shape[-1], pred_audio.shape[-1])
            audio = audio[..., :min_len]
            pred_audio = pred_audio[..., :min_len]
        return self.metric(pred_audio, audio)
