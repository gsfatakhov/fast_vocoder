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
            with torch.no_grad():
                outputs = self.model(**batch)

            batch.update(outputs)
            all_losses = self.criterion(compute_generator_loss =False, **batch)
            batch.update(all_losses)

            self.disc_optimizer.zero_grad()
            batch["loss_disc"].backward()
            self._clip_grad_norm()
            self.disc_optimizer.step()
            if self.disc_lr_scheduler is not None:
                self.disc_lr_scheduler.step()

            # обучение генератора
            outputs = self.model(**batch)
            batch.update(outputs)
            all_losses = self.criterion(compute_discriminator_loss=False, **batch)
            batch.update(all_losses)

            self.gen_optimizer.zero_grad()
            batch["loss_gen"].backward()
            self._clip_grad_norm()
            self.gen_optimizer.step()
            if self.gen_lr_scheduler is not None:
                self.gen_lr_scheduler.step()

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

            # TODO may be visualize mel spectrograms
            # self.writer.add_image("mel_real", mel_real)
            # self.writer.add_image("mel_pred", mel_pred)

    def _log_audio(self, batch):
        self.writer.add_audio("audio_first", batch["audio"][0], 22050)
        self.writer.add_audio("pred_audio_first", batch["pred_audio"][0], 22050)

        i = random.randint(0, batch["audio"].size(0) - 1)
        self.writer.add_audio("audio_random", batch["audio"][i], 22050)
        self.writer.add_audio("pred_audio_random", batch["pred_audio"][i], 22050)
