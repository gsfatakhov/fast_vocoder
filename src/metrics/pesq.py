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

        metric_fs = fs
        self.need_resample = False

        if fs not in [8000, 16000]:
            if fs > 16000:
                metric_fs = 16000
            else:
                metric_fs = 8000

            self.resampler = Resample(fs, metric_fs).to(device)
            self.need_resample = True

        self.metric = PerceptualEvaluationSpeechQuality(metric_fs, mode=mode, n_processes=n_processes).to(device)

    def __call__(
            self,
            audio: torch.Tensor,
            pred_audio: torch.Tensor,
            **kwargs
    ):
        if self.need_resample:
            audio = self.resampler(audio)
            pred_audio = self.resampler(pred_audio)
        val = 0.0
        try:
            val = self.metric(pred_audio, audio)
        except NoUtterancesError:
            pass
        return val
