import warnings
import hydra
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

from src.datasets.custom_dir_dataset import CustomDirDataset
from src.trainer.synthesis_inferencer import SynthesisInferencer

warnings.filterwarnings("ignore", category=UserWarning)

@hydra.main(version_base=None, config_path="src/configs", config_name="synthesis")
def main(config):
    set_random_seed(config.synthesizer.seed)

    if config.synthesizer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.synthesizer.device

    tacotron2 = torch.hub.load(
        'NVIDIA/DeepLearningExamples:torchhub',
        'nvidia_tacotron2',
        pretrained=False
    )

    checkpoint = torch.hub.load_state_dict_from_url(
        'https://api.ngc.nvidia.com/v2/models/nvidia/'
        'tacotron2pyt_fp32/versions/1/files/'
        'nvidia_tacotron2pyt_fp32_20190306.pth',
        map_location=torch.device(device)
    )

    state_dict = {
        k.replace('module.', ''): v
        for k, v in checkpoint['state_dict'].items()
    }
    tacotron2.load_state_dict(state_dict)

    tacotron2.decoder.max_decoder_steps = config.synthesizer.tacotron_max_decoder_steps
    tacotron2.eval()

    model = instantiate(config.model).to(device)
    print(model)

    if config.synthesizer.text is not None:
        data = [{"audio_name": "single_utterance", "text": config.synthesizer.text}]
        dataloader = DataLoader(data, batch_size=1, shuffle=False)
    else:
        dataset = CustomDirDataset(config.data.root_dir)
        dataloader = DataLoader(dataset, batch_size=config.data.batch_size, num_workers=config.data.num_workers, shuffle=False)

    dataloaders = {"inference": dataloader}

    metrics = None
    if "metrics" in config and config.metrics is not None:
        metrics = {"inference": [instantiate(config.metrics)]}

    save_path = ROOT_PATH / config.synthesizer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer = SynthesisInferencer(
        model=model,
        tacotron2=tacotron2,
        config=config,
        device=device,
        dataloaders=dataloaders,
        save_path=save_path,
        metrics=metrics,
        batch_transforms=None,
        skip_model_load=(config.synthesizer.from_pretrained is None),
    )

    logs = inferencer.run_inference()

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")
    print("Results saved to:", save_path)

if __name__ == "__main__":
    main()
