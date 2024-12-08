import torch

def collate_fn(dataset_items: list[dict]):

    audios = [item["audio"] for item in dataset_items]
    texts = [item["text"] for item in dataset_items]
    audio_names = [item["audio_name"] for item in dataset_items]


    audio_batch = torch.stack(audios, dim=0)  # [B, 1, T]

    if "mel" not in dataset_items[0]:
        return {
            "audio": audio_batch,
            "text": texts,
            "audio_name": audio_names
        }

    mels = [item["mel"] for item in dataset_items]
    mels_batch = torch.stack(mels, dim=0).squeeze(1)  # [B, n_m

    return {
        "audio": audio_batch,
        "text": texts,
        "audio_name": audio_names,
        "mel": mels_batch
    }
