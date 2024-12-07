import torch
import torch.nn.functional as F
from src.utils.mel import MelSpectrogram, MelSpectrogramConfig

def collate_fn(dataset_items: list[dict]):
    """
    Функция коллейта для LJSpeech датасета.
    Ставит вместе несколько элементов датасета в батч,
    дополняя аудио по длине до максимальной длины в батче.

    Args:
        dataset_items (list[dict]): список элементов из датасета LJSpeechDataset.

    Returns:
        dict: словарь с полями:
            "audio" (Tensor): батч аудио [B, 1, T]
            "text" (list[str]): список текстов
            "audio_name" (list[str]): список имён аудио
    """

    audios = [item["audio"] for item in dataset_items]
    texts = [item["text"] for item in dataset_items]
    audio_names = [item["audio_name"] for item in dataset_items]

    # Определяем максимальную длину аудио в батче
    max_len = max([a.shape[-1] for a in audios])

    # Дополняем аудио нулями до максимальной длины
    audios_padded = [F.pad(a, (0, max_len - a.shape[-1])) for a in audios]

    # Ставим батч вместе
    audio_batch = torch.stack(audios_padded, dim=0)  # [B, 1, T]

    # Создаем мел-спектрограммы

    mel_config = MelSpectrogramConfig()
    mel_extractor = MelSpectrogram(mel_config)

    mels = mel_extractor(audio_batch.squeeze(1))  # [B, n_mels, T']

    # Check mels lentghs equal to audio lengths
    # assert audio_batch.shape[-1] == mels.shape[-1]


    return {
        "audio": audio_batch,
        "text": texts,
        "audio_name": audio_names,
        "mel": mels
    }
