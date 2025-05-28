import torchaudio
from tqdm.auto import tqdm
from pathlib import Path

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json

import torch
import torch.nn.functional as F

from src.utils.mel import MelSpectrogram

MAX_WAV_VALUE = 32768.0


class LJSpeechDataset(BaseDataset):
    def __init__(
            self,
            name="train",
            target_sr=22050,
            audio_path=ROOT_PATH / "data" / "ljspeech",
            index_audio_path=None,
            use_normalized_text=True,
            segment_length=8192,
            mel_config=None,
            calc_mel_for_loss=False,
            fmax_loss=None,
            n_cache_reuse=0,
            *args, **kwargs
    ):
        self.name = name
        self.target_sr = target_sr
        self.audio_path = Path(audio_path)
        self.use_normalized_text = use_normalized_text
        self.segment_length = segment_length

        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0

        self.mel_config = mel_config
        self.calc_mel_for_loss = calc_mel_for_loss
        if self.mel_config:
            self.mel_extractor = MelSpectrogram(self.mel_config)

            if self.calc_mel_for_loss:
                self.loss_mel_config = self.mel_config
                self.loss_mel_config.f_max = fmax_loss

                self.loss_mel_extractor = MelSpectrogram(self.loss_mel_config)

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

    @staticmethod
    def _normalize_tensor(tensor, fill_value=0.0):
        """
        Analog of librosa.util.normalize
        """
        if not torch.all(torch.isfinite(tensor)):
            return torch.full_like(tensor, fill_value)

        max_val = torch.abs(tensor).max()
        if max_val > 0:
            return tensor / max_val
        else:
            return torch.zeros_like(tensor)

    def __getitem__(self, ind):
        data_dict = self._index[ind]

        audio_path = data_dict["audio_path"]
        text = data_dict["text"]
        audio_name = data_dict["audio_name"]

        if self._cache_ref_count == 0:
            audio_tensor = self.load_audio(audio_path)
            audio_tensor = audio_tensor / MAX_WAV_VALUE

            # TODO: check if possible to norm in transforms
            audio_tensor = self._normalize_tensor(audio_tensor) * 0.95

            self.cached_wav = audio_tensor
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio_tensor = self.cached_wav
            self._cache_ref_count -= 1

        if self.name == "train":
            if audio_tensor.shape[-1] >= self.segment_length:
                if self.mel_config:
                    # length without padding for attn
                    length_mel = torch.LongTensor([ self.segment_length // self.mel_config.hop_length])
                max_start = audio_tensor.shape[-1] - self.segment_length
                start = torch.randint(0, max_start + 1, (1,)).item()
                audio_tensor = audio_tensor[:, start:start + self.segment_length]
            else:
                if self.mel_config:
                    length_mel = torch.LongTensor([audio_tensor.shape[-1] // self.mel_config.hop_length])
                diff = self.segment_length - audio_tensor.shape[-1]
                audio_tensor = F.pad(audio_tensor, (0, diff))

        else:
            # 21 seconds is the longest audio in the test dataset, sr=22050 n = 57
            test_length = self.segment_length
            if test_length == None:
                # pad to max length
                test_length = audio_tensor.shape[-1]
            if audio_tensor.shape[-1] > test_length:
                if self.mel_config:
                    # length without padding for attn
                    length_mel = torch.LongTensor([test_length // self.mel_config.hop_length])
                audio_tensor = audio_tensor[:, :test_length]
            else:
                if self.mel_config:
                    length_mel = torch.LongTensor([audio_tensor.shape[-1] // self.mel_config.hop_length])
                diff = test_length - audio_tensor.shape[-1]
                audio_tensor = F.pad(audio_tensor, (0, diff))

        instance_data = {
            "audio": audio_tensor,
            "text": text,
            "audio_name": audio_name,
        }

        if self.mel_config:
            instance_data["mel"] = self.mel_extractor(audio_tensor)  # [B, n_mels, T']
            instance_data["length"] = length_mel
            instance_data["wav_length"] = audio_tensor.shape[-1] / self.target_sr
            if self.calc_mel_for_loss:
                instance_data["mel_for_loss"] = self.loss_mel_extractor(audio_tensor)

        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def load_audio(self, path):
        # audio_tensor shape [C, T]
        audio_tensor, sr = torchaudio.load(path)
        # means to [1, T]
        audio_tensor = audio_tensor.mean(dim=0, keepdim=True)

        if sr != self.target_sr:
            # audio_tensor = torchaudio.functional.resample(audio_tensor, sr, self.target_sr)
            raise ValueError("{} SR doesn't match target {} SR".format(sr, self.target_sr))
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
