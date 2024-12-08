import torch
import torchaudio
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer для задачи ресинтеза/синтеза речи.
    Вместо логитов и лейблов, работаем с аудио.
    """

    def __init__(
            self,
            model,
            config,
            device,
            dataloaders,
            save_path,
            metrics=None,
            batch_transforms=None,
            skip_model_load=False,
    ):
        """
        Args:
            model (nn.Module): модель (например, HiFiGAN).
            config (DictConfig): конфигурация ран-а.
            device (str): устройство для тензоров и модели ("cpu" или "cuda").
            dataloaders (dict[DataLoader]): Даталоадеры для разных частей (val, test и т.д.).
            save_path (str): путь для сохранения результатов инференса.
            metrics (dict): словарь метрик для оценки качества (например PESQ).
            batch_transforms (dict[nn.Module] | None): трансформации для всего батча.
            skip_model_load (bool): если False, требуется путь к чекпойнту;
                                    если True, модель уже загружена снаружи.
        """
        assert (
                skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        self.save_path = save_path

        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Запустить инференс на каждом датасете, вернуть логи.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Прогоняем батч через модель, считаем метрики, сохраняем результаты.

        Предполагается, что batch имеет поля:
        - "mel": [B, n_mels, T_mel]
        - "audio": [B, 1, T]

        Модель добавит "pred_audio": [B, 1, T'] в batch.

        Сохраняем результаты в wav формат.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device

        # Прогоняем модель
        outputs = self.model(**batch)
        batch.update(outputs)

        # Считаем метрики, если есть
        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        # Сохраняем результаты
        # Создадим уникальный ID для каждого сэмпла
        batch_size = batch["audio"].shape[0]
        current_id = batch_idx * batch_size

        # Предполагается, что мы хотим сохранить предсказанное аудио
        # в виде wav файлов. Чтобы сохранить, нужно знать sample_rate.
        # Допустим, используем mel-config sr или вы храните sr в self.config.
        sr = 22050

        for i in range(batch_size):
            pred_audio = batch["pred_audio"][i].clone().detach().cpu().squeeze(0)  # [T]

            # Сохраняем wav
            if self.save_path is not None:
                out_path = self.save_path / part / f"{batch["audio_name"][i]}.wav"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(str(out_path), pred_audio.unsqueeze(0), sr)

        return batch

    def _inference_part(self, part, dataloader):
        """
        Запуск инференса на заданной части датасета (например, 'val' или 'test').

        Возвращаем словарь метрик по итогам инференса.
        """
        self.is_train = False
        self.model.eval()

        if self.evaluation_metrics is not None:
            self.evaluation_metrics.reset()

        # Создаем папку для сохранения результатов
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        if self.evaluation_metrics is not None:
            return self.evaluation_metrics.result()
        else:
            return {}
