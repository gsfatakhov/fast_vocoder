import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from src.utils.mel import MelSpectrogram, MelSpectrogramConfig

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    logger.info(model)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function, model=model).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    gen_trainable_params = filter(lambda p: p.requires_grad, model.generator.parameters())
    gen_optimizer = instantiate(config.gen_optimizer, params=gen_trainable_params)
    gen_lr_scheduler = instantiate(config.gen_lr_scheduler, optimizer=gen_optimizer)

    disc_trainable_params = filter(lambda p: p.requires_grad, model.discriminator.parameters())
    disc_optimizer = instantiate(config.disc_optimizer, params=disc_trainable_params)
    disc_lr_scheduler = instantiate(config.disc_lr_scheduler, optimizer=disc_optimizer)

    #TODO: make it global params
    mel_extractor = instantiate(config.mel_extractor).to(device)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        gen_optimizer=gen_optimizer,
        gen_lr_scheduler=gen_lr_scheduler,
        disc_optimizer=disc_optimizer,
        disc_lr_scheduler=disc_lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
        mel_extractor=mel_extractor,
    )

    trainer.train()


if __name__ == "__main__":
    main()
