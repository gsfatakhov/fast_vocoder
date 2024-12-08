import torch

from src.metrics.base_metric import BaseMetric
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility


class STOI(BaseMetric):
    def __init__(self, device="auto", fs=22050, extended=False, *args, **kwargs):
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
        # nuber of positive values in audios
        # pos = sum([torch.sum(audio > 0).item() for audio in audio])
        # neg = sum([torch.sum(audio < 0).item() for audio in audio])

        # print("Last Positive values in audios:", pos)
        # print("Last Negative values in audios:", neg)

        # make audio_ref and pred_audio same T
        # if audio.shape[-1] != pred_audio.shape[-1]:
        #     min_len = min(audio.shape[-1], pred_audio.shape[-1])
        #     audio = audio[..., :min_len]
        #     pred_audio = pred_audio[..., :min_len]

        # Ensure at least 30 frames for intermediate intelligibility
        # if audio.shape[-1] < 30:
        #     print(audio.shape[-1])
        #     raise ValueError(
        #         "Not enough STFT frames to compute intermediate intelligibility measure after removing silent frames. "
        #         "Please check your audio files."
        #     )

        # shapes

        # print("audio shape:", audio.shape)
        # print("pred_audio shape:", pred_audio.shape)

        #ensure audio and pred_audio between -1 and 1
        # audio = torch.clamp(audio, -1, 1)
        # pred_audio = torch.clamp(pred_audio, -1, 1)

        # print(audio[0])
        # print(pred_audio[0])

        return self.metric(pred_audio, audio)
