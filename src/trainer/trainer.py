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

            # 1. Обучение дискриминатора:
            # Прогоняем модель в режиме detach для предсказаний (или внутри лосса учитываем detach)
            with torch.no_grad():
                outputs = self.model(**batch)  # pred_audio получаем

            batch.update(outputs)
            all_losses = self.criterion(**batch)  # здесь лосс считается с detach для фейковых предсказаний
            batch.update(all_losses)

            self.disc_optimizer.zero_grad()
            batch["loss_disc"].backward()
            self._clip_grad_norm()
            self.disc_optimizer.step()
            if self.disc_lr_scheduler is not None:
                self.disc_lr_scheduler.step()

            # 2. Обучение генератора:
            # Заново прогоняем модель, чтобы построить новый граф без detach
            outputs = self.model(**batch)
            batch.update(outputs)
            all_losses = self.criterion(**batch)
            batch.update(all_losses)

            self.gen_optimizer.zero_grad()
            batch["loss_gen"].backward()
            self._clip_grad_norm()
            self.gen_optimizer.step()
            if self.gen_lr_scheduler is not None:
                self.gen_lr_scheduler.step()

        else:
            # В режиме валидации:
            outputs = self.model(**batch)
            batch.update(outputs)
            all_losses = self.criterion(**batch)
            batch.update(all_losses)

        # Логирование метрик и лоссов
        for loss_name in self.config.writer.loss_names:
            if loss_name in batch:
                metrics.update(loss_name, batch[loss_name].item())
        # if metric_funcs:
        for met in metric_funcs:
            metrics.update(met.name, met(**batch))

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            self._log_audio(batch)

            # TODO visualize mel spectrograms
            # self.writer.add_image("mel_real", mel_real)
            # self.writer.add_image("mel_pred", mel_pred)

    def _log_audio(self, batch):
        self.writer.add_audio("audio_first", batch["audio"][0], 22050)
        self.writer.add_audio("pred_audio_first", batch["pred_audio"][0], 22050)

        i = random.randint(0, batch["audio"].size(0) - 1)
        self.writer.add_audio("audio_random", batch["audio"][i], 22050)
        self.writer.add_audio("pred_audio_random", batch["pred_audio"][i], 22050)
