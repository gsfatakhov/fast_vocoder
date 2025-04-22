import torch

from src.metrics.base_metric import BaseMetric
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from pesq import NoUtterancesError
from torchaudio.transforms import Resample


class PESQ(BaseMetric):
    def __init__(self, device="auto", fs=22050, mode="wb", n_processes=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.orig_fs = fs
        self.need_resample = False
        if self.orig_fs > 16000:
            self.resampler = Resample(self.orig_fs, 16000)
            self.need_resample = True
        elif self.orig_fs != 8000:
            self.resampler = Resample(self.orig_fs, 8000)
            self.need_resample = True

        self.metric = PerceptualEvaluationSpeechQuality(fs, mode=mode, n_processes=n_processes).to(device)

    def __call__(
            self,
            audio: torch.Tensor,
            pred_audio: torch.Tensor,
            **kwargs
    ):
        if self.need_resample:
            audio = self.resampler
        val = 0.0
        try:
            val = self.metric(pred_audio, audio)
        except NoUtterancesError:
            pass
        return val
