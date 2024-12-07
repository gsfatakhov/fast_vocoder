import torchaudio
from tqdm.auto import tqdm
from pathlib import Path

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class LJSpeechDataset(BaseDataset):
    """
    LJSpeech dataset loader, following a structure similar to AvssDataset.
    The dataset is expected to be organized as follows:

    <audio_path>/
        LJSpeech-1.1/
            metadata.csv
            wavs/
                LJ001-0002.wav
                LJ001-0003.wav
                ...

    The metadata.csv file contains lines in the format:
    LJ001-0002|in being comparatively modern.|in being comparatively modern.
    where:
        - The first field is the file ID (e.g., LJ001-0002).
        - The second field is normalized text.
        - The third field is the original text.

    This dataset class will read `metadata.csv`, create an index mapping each
    file ID to its corresponding audio and text, load and preprocess audio,
    and return it along with the corresponding text.
    """

    def __init__(
            self,
            name="train",
            target_sr=22050,
            audio_path=ROOT_PATH / "data" / "ljspeech",
            index_audio_path=None,
            use_normalized_text=True,
            *args, **kwargs
    ):
        """
        Args:
            name (str): partition name. Since LJSpeech doesn't have predefined splits,
                         you may define your own splits externally and provide a custom
                         metadata or subset. For now, will assume the entire dataset as 'train'.
            target_sr (int): target sample rate for loading audio files.
            audio_path (Path): path to the directory containing LJSpeech-1.1 folder.
            index_audio_path (Path): path where to store or read the index.json file.
            use_normalized_text (bool): if True, use normalized text (2nd field from metadata),
                                        else use the original text (3rd field).
        """
        self.name = name
        self.target_sr = target_sr
        self.audio_path = Path(audio_path)
        self.use_normalized_text = use_normalized_text

        if not index_audio_path:
            index_audio_path = self.audio_path
        else:
            index_audio_path = Path(index_audio_path)
        self.index_audio_path = index_audio_path / name
        self.index_audio_path.mkdir(exist_ok=True, parents=True)

        self.index_audio_path /= "index.json"

        # Check if index already exists, otherwise create it
        if self.index_audio_path.exists():
            index = read_json(str(self.index_audio_path))
        else:
            index = self._create_index()

        super().__init__(index, *args, **kwargs)

    def __getitem__(self, ind):
        """
        Get element from the index, load and preprocess it.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict):
                {
                    "audio": torch.Tensor,
                    "text": str,
                    "audio_name": str
                }
        """
        data_dict = self._index[ind]

        audio_path = data_dict["audio_path"]
        text = data_dict["text"]
        audio_name = data_dict["audio_name"]

        audio_tensor = self.load_audio(audio_path)

        instance_data = {
            "audio": audio_tensor,
            "text": text,
            "audio_name": audio_name
        }

        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # keep only the first channel
        target_sr = self.target_sr
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def _create_index(self):
        """
        Parse metadata.csv to create the dataset index.
        Returns:
            index (list[dict]): each entry contains:
                {
                    "audio_path": str,
                    "text": str,
                    "audio_name": str
                }
        """
        index = []
        metadata_path = self.audio_path / "metadata.csv"
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
                # Example line format:
                # LJ001-0002|in being comparatively modern.|in being comparatively modern.
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

        # write index to disk
        write_json(index, self.index_audio_path)
        return index

    def _assert_index_is_valid(self, index):
        for entry in index:
            assert "audio_path" in entry, "Missing field 'audio_path'"
            assert "text" in entry, "Missing field 'text'"
            assert "audio_name" in entry, "Missing field 'audio_name'"