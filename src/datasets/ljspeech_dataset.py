import torchaudio
from tqdm.auto import tqdm
from pathlib import Path

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json

import torch
import torch.nn.functional as F

from src.utils.mel import MelSpectrogram, MelSpectrogramConfig


class LJSpeechDataset(BaseDataset):
    def __init__(
            self,
            name="train",
            target_sr=22050,
            audio_path=ROOT_PATH / "data" / "ljspeech",
            index_audio_path=None,
            use_normalized_text=True,
            segment_length=8191,
            calc_mel=True,
            *args, **kwargs
    ):
        self.name = name
        self.target_sr = target_sr
        self.audio_path = Path(audio_path)
        self.use_normalized_text = use_normalized_text
        self.segment_length = segment_length

        self.calc_mel = calc_mel
        if calc_mel:
            self.mel_config = MelSpectrogramConfig()
            self.mel_extractor = MelSpectrogram(self.mel_config)

        if not index_audio_path:
            index_audio_path = self.audio_path
        else:
            index_audio_path = Path(index_audio_path)
        self.index_audio_path = index_audio_path / name
        self.index_audio_path.mkdir(exist_ok=True, parents=True)

        self.index_audio_path /= "index.json"

        if self.index_audio_path.exists():
            index = read_json(str(self.index_audio_path))
        else:
            index = self._create_index()

        super().__init__(index, *args, **kwargs)

    def __getitem__(self, ind):
        data_dict = self._index[ind]

        audio_path = data_dict["audio_path"]
        text = data_dict["text"]
        audio_name = data_dict["audio_name"]

        audio_tensor = self.load_audio(audio_path)

        if self.name == "train":
            if audio_tensor.shape[-1] >= self.segment_length:
                max_start = audio_tensor.shape[-1] - self.segment_length
                start = torch.randint(0, max_start + 1, (1,)).item()
                audio_tensor = audio_tensor[:, start:start + self.segment_length]
            else:
                diff = self.segment_length - audio_tensor.shape[-1]
                audio_tensor = F.pad(audio_tensor, (0, diff))

        else:
            # 21 seconds is the longest audio in the test dataset, sr=22050 n = 57
            test_length = self.segment_length
            if audio_tensor.shape[-1] > test_length:
                audio_tensor = audio_tensor[:, :test_length]
            else:
                diff = test_length - audio_tensor.shape[-1]
                audio_tensor = F.pad(audio_tensor, (0, diff))



        instance_data = {
            "audio": audio_tensor,
            "text": text,
            "audio_name": audio_name,
        }

        if self.calc_mel:
            instance_data["mel"] = self.mel_extractor(audio_tensor)  # [B, n_mels, T']

        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        # Берем только один канал
        audio_tensor = audio_tensor[0:1, :]
        target_sr = self.target_sr
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def _create_index(self):
        index = []
        metadata_path = self.audio_path / (self.name + "_metadata.csv")
        wavs_path = self.audio_path / "wavs"

        assert metadata_path.exists(), f"metadata.csv not found at {metadata_path}"
        assert wavs_path.exists(), f"wavs folder not found at {wavs_path}"

        print(f"Creating LJSpeech Dataset ({self.name})...")
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue

                parts = line.split("|")
                if len(parts) < 3:
                    continue
                file_id = parts[0].strip()
                normalized_text = parts[1].strip()
                original_text = parts[2].strip() if len(parts) > 2 else normalized_text

                text = normalized_text if self.use_normalized_text else original_text
                audio_file = file_id + ".wav"
                audio_path = wavs_path / audio_file
                if not audio_path.exists():
                    continue

                index.append({
                    "audio_path": str(audio_path),
                    "text": text,
                    "audio_name": file_id
                })

        write_json(index, self.index_audio_path)
        return index

    def _assert_index_is_valid(self, index):
        for entry in index:
            assert "audio_path" in entry, "Missing field 'audio_path'"
            assert "text" in entry, "Missing field 'text'"
            assert "audio_name" in entry, "Missing field 'audio_name'"
