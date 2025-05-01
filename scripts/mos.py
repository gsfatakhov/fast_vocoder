from torch.serialization import add_safe_globals
from wvmos import get_wvmos


add_safe_globals(['pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint'])

model = get_wvmos()
mos = model.calculate_dir("/Users/georgijfatahov/PycharmProjects/fast_vocoder/resynthesize_results", mean=True)
print(mos)
