from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
import torch
import random


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

            # Для дискриминатора получаем генерации без градиентов
            self.disc_optimizer.zero_grad()
            with torch.no_grad():
                outputs = self.model(**batch)
            batch.update(outputs)

            disc_loss = self.criterion(compute_generator_loss=False, **batch)
            batch.update(disc_loss)

            batch["loss_disc"].backward()
            self._clip_grad_norm()
            self.disc_optimizer.step()

            # обучение генератора
            for param in self.model.discriminator.parameters():
                param.requires_grad = False
            self.gen_optimizer.zero_grad()

            outputs = self.model(**batch)
            batch.update(outputs)
            gen_loss = self.criterion(compute_discriminator_loss=False, **batch)
            batch.update(gen_loss)

            batch["loss_gen"].backward()
            self._clip_grad_norm()
            self.gen_optimizer.step()

            for param in self.model.discriminator.parameters():
                param.requires_grad = True
        else:
            # eval
            outputs = self.model(**batch)
            batch.update(outputs)
            all_losses = self.criterion(**batch)
            batch.update(all_losses)

        for loss_name in self.config.writer.loss_names:
            if loss_name in batch:
                metrics.update(loss_name, batch[loss_name].item())
        # if metric_funcs:
        for met in metric_funcs:
            metrics.update(met.name, met(**batch))

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            self._log_audio(batch)
            self._log_mel(batch)

    def _log_mel(self, batch):
        if "pred_mel" not in batch:
            mel_pred = self.mel_extractor(batch["pred_audio"][0])
        else:
            mel_pred = batch["pred_mel"][0]

        self.writer.add_image("mel_real_first", batch["mel"][0])
        self.writer.add_image("mel_pred_first", mel_pred)

    def _log_audio(self, batch):
        self.writer.add_audio("audio_first", batch["audio"][0], 22050)
        self.writer.add_audio("pred_audio_first", batch["pred_audio"][0], 22050)

        i = random.randint(0, batch["audio"].size(0) - 1)
        self.writer.add_audio("audio_random", batch["audio"][i], 22050)
        self.writer.add_audio("pred_audio_random", batch["pred_audio"][i], 22050)
