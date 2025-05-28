import torch


def collate_fn(dataset_items: list[dict]):
    audios = [item["audio"] for item in dataset_items]
    texts = [item["text"] for item in dataset_items]
    audio_names = [item["audio_name"] for item in dataset_items]

    audio_batch = torch.stack(audios, dim=0)  # [B, 1, T]

    out = {
        "audio": audio_batch,
        "text": texts,
        "audio_name": audio_names
    }

    if "mel" in dataset_items[0]:
        mels = [item["mel"] for item in dataset_items]
        out["mel"] = torch.stack(mels, dim=0).squeeze(1)  # [B, n_m, T]

        lengths = [item["length"] for item in dataset_items]
        out["length"] = torch.stack(lengths, dim=0).squeeze(1)

        if "mel_for_loss" in dataset_items[0]:
            mels_for_loss = [item["mel_for_loss"] for item in dataset_items]
            out["mel_for_loss"] = torch.stack(mels_for_loss, dim=0).squeeze(1)

        out["wav_length"] = dataset_items[0]["wav_length"]

    return out
