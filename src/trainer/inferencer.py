import torch
import torchaudio
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer

import time


class Inferencer(BaseTrainer):
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
            warmup_n=0,
    ):
        assert (
                skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

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

        self.warmup_n = warmup_n
        self.warmup_counter = 0

        self.gen_time = 0
        self.inp_time = 0

    def run_inference(self):
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            logs["RTFX"] = self.inp_time / self.gen_time
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        batch_size = batch["audio"].shape[0]

        if self.warmup_counter >= self.warmup_n:
            start = time.perf_counter()
            outputs = self.model(**batch)
            end = time.perf_counter()

            self.gen_time += end - start
            self.inp_time += batch["wav_length"] * batch_size

        else:
            self.warmup_counter += 1
            outputs = self.model(**batch)

        batch.update(outputs)

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        sr = 22050

        for i in range(batch_size):
            pred_audio = batch["pred_audio"][i].clone().detach().cpu().squeeze(0)  # [T]

            if self.save_path is not None:
                out_path = self.save_path / part / f"{batch['audio_name'][i]}.wav"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(str(out_path), pred_audio.unsqueeze(0), sr)

        return batch

    def _inference_part(self, part, dataloader):
        self.is_train = False
        self.model.eval()

        if self.evaluation_metrics is not None:
            self.evaluation_metrics.reset()

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
