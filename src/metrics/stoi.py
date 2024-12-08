import torch

from src.metrics.base_metric import BaseMetric
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility


class STOI(BaseMetric):
    def __init__(self, device="auto", fs=22050, extended=False, group_size=3, *args, **kwargs):
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
        self.group_size = group_size

    def __call__(self, audio: torch.Tensor, pred_audio: torch.Tensor, **kwargs):
        """
        Compute STOI between predicted and reference audio using concatenation.

        Args:
            audio (Tensor): Reference audio [B, 1, T].
            pred_audio (Tensor): Predicted audio [B, 1, T].

        Returns:
            (float): STOI score averaged over the batch.
        """
        batch_size, _, total_length = audio.shape
        stoi_scores = []

        for i in range(0, batch_size, self.group_size):
            audio_group = audio[i:i + self.group_size]
            pred_audio_group = pred_audio[i:i + self.group_size]

            audio_combined = torch.cat([x.squeeze(0) for x in audio_group], dim=-1)
            pred_audio_combined = torch.cat([x.squeeze(0) for x in pred_audio_group], dim=-1)

            audio_combined = audio_combined / torch.max(torch.abs(audio_combined))
            pred_audio_combined = pred_audio_combined / torch.max(torch.abs(pred_audio_combined))

            stoi_value = self.metric(pred_audio_combined.unsqueeze(0), audio_combined.unsqueeze(0))
            stoi_scores.append(stoi_value.item())

        return sum(stoi_scores) / len(stoi_scores)