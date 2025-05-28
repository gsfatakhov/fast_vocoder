from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
try:
    from torch.serialization import add_safe_globals
except ImportError:
    def add_safe_globals(globals_dict):
        return globals_dict

from wvmos import get_wvmos

add_safe_globals([ModelCheckpoint])

model = get_wvmos()
mos = model.calculate_dir("/home/gsfatakhov/fast_vocoder/LJSpeech-1.1/wavs", mean=True)
print(mos)
