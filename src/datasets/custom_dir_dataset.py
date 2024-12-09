from pathlib import Path
from torch.utils.data import Dataset

class CustomDirDataset(Dataset):
    def __init__(self, root: str, *args, **kwargs):
        self.root = Path(root)
        self.transcriptions_dir = self.root / "transcriptions"
        assert self.transcriptions_dir.exists(), f"No transcriptions folder found in {self.root}"

        self.index = self._create_index()

    def _create_index(self):
        index = []
        for fname in sorted(self.transcriptions_dir.glob("*.txt")):
            utterance_id = fname.stem
            with open(fname, "r", encoding="utf-8") as f:
                text = f.read().strip()
            index.append({
                "audio_name": utterance_id,
                "text": text
            })
        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return self.index[idx]
