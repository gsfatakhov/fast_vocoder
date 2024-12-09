import torch
import torchaudio
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer

class SynthesisInferencer(BaseTrainer):
    """
    Inferencer for pipline: text -> mel (Tacotron2) -> audio (HiFiGAN)
    """

    def __init__(
            self,
            model,
            tacotron2,
            config,
            device,
            dataloaders,
            save_path,
            metrics=None,
            batch_transforms=None,
            skip_model_load=False,
    ):
        assert (
            skip_model_load or config.synthesizer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.synthesizer

        self.device = device

        self.model = model.to(device)
        self.batch_transforms = batch_transforms

        self.tacotron2 = tacotron2.to(device)
        self.tacotron2.eval()

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
            self._from_pretrained(config.synthesizer.get("from_pretrained"))

    def run_inference(self):
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        batch = self.move_batch_to_device(batch)
        # batch = self.transform_batch(batch)

        texts = batch["text"]
        audio_names = batch["audio_name"]
        sr = 22050

        pred_audios = []

        with torch.no_grad():
            for i, text in enumerate(texts):
                from tacotron2.text import text_to_sequence
                sequence = torch.LongTensor(text_to_sequence(text, ['english_cleaners']))[None, :].to(self.device)
                input_lengths = torch.IntTensor([sequence.shape[1]]).to(self.device)

                # [1, n_mels, T_mel]
                mel, _, _ = self.tacotron2.infer(sequence, input_lengths)

                voc_batch = {"mel": mel}
                voc_batch = self.model(**voc_batch)
                pred_audio = voc_batch["pred_audio"].cpu()  # [1, 1, T]

                pred_audios.append(pred_audio)

        pred_audios = torch.cat(pred_audios, dim=0)  # [B, 1, T]
        batch["pred_audio"] = pred_audios

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        batch_size = pred_audios.shape[0]

        if self.save_path is not None:
            for i in range(batch_size):
                out_path = self.save_path / part / f"{audio_names[i]}.wav"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(str(out_path), pred_audios[i], sr)

        return batch

    def _inference_part(self, part, dataloader):
        self.is_train = False
        self.model.eval()
        self.tacotron2.eval()

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
