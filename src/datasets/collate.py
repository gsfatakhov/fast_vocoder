import torch
import torch.nn.functional as F
from src.utils.mel import MelSpectrogram, MelSpectrogramConfig

def collate_fn(dataset_items: list[dict]):
    """
    Функция коллейта для LJSpeech датасета.
    Предполагается, что dataset уже возвращает аудио сегменты фиксированной длины segment_length.
    Нам остаётся их просто склеить в батч, и посчитать мел-спектры.
    """

    audios = [item["audio"] for item in dataset_items]
    texts = [item["text"] for item in dataset_items]
    audio_names = [item["audio_name"] for item in dataset_items]

    # count positive and negative amplitudes in all audios
    # nuber of positive values in audios
    # pos = sum([torch.sum(audio > 0).item() for audio in audios])
    # neg = sum([torch.sum(audio < 0).item() for audio in audios])
    # print("Positive values in audios:", pos)
    # print("Negative values in audios:", neg)


    audio_batch = torch.stack(audios, dim=0)  # [B, 1, T]

    # Создаем мел-спектрограммы

    mel_config = MelSpectrogramConfig()
    mel_extractor = MelSpectrogram(mel_config)

    mels = mel_extractor(audio_batch.squeeze(1))  # [B, n_mels, T']

    return {
        "audio": audio_batch,
        "text": texts,
        "audio_name": audio_names,
        "mel": mels
    }
