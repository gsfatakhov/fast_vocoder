import torch
from sympy import preorder_traversal

from src.metrics.base_metric import BaseMetric
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality


class PESQ(BaseMetric):
    # Не поддерживает fs=22050
    def __init__(self, device="auto", fs=16000, mode="wb", n_processes=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = PerceptualEvaluationSpeechQuality(fs, mode=mode, n_processes=n_processes).to(device)

    def __call__(
            self,
            audio: torch.Tensor,
            pred_audio: torch.Tensor,
            **kwargs
    ):
        return self.metric(pred_audio, audio)
