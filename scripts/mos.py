from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.serialization import add_safe_globals
from wvmos import get_wvmos

add_safe_globals([ModelCheckpoint])

model = get_wvmos()
mos = model.calculate_dir("/home/georgii/PycharmProjects/fast_vocoder/datasets/5_ljspeech_dataset/wavs", mean=True)
print(mos)
