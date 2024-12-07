from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
import torch


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

            self.desc_optimizer.zero_grad()
            batch["loss_disc"].backward()
            self._clip_grad_norm()
            self.desc_optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

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
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        else:
            # В режиме валидации:
            outputs = self.model(**batch)
            batch.update(outputs)
            all_losses = self.criterion(self.model, **batch)
            batch.update(all_losses)

        # Логирование метрик и лоссов
        for loss_name in self.config.writer.loss_names:
            if loss_name in batch:
                metrics.update(loss_name, batch[loss_name].item())

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
            pass
