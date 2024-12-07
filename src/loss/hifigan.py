import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.mel import MelSpectrogram, MelSpectrogramConfig


class HiFiGANLoss(nn.Module):
    """
    HiFi-GAN Loss:
    Возвращает словарь с лоссами:
    - "loss_gen": Лосс для генератора (адверсариальный + feature matching + mel loss)
    - "loss_disc": Лосс для дискриминатора
    - "loss_mel": mel-спектральный лосс (входит в loss_gen)
    - "loss_fm": feature matching лосс (входит в loss_gen)
    - "loss_adv_gen": advarsarialный лосс генератора (входит в loss_gen)
    """

    def __init__(self, mel_config=MelSpectrogramConfig()):
        super().__init__()
        self.mel_extractor = MelSpectrogram(mel_config)
        self.l1 = nn.L1Loss()

    def forward(self, model, **batch):
        """
        Args:
            model: HiFiGAN модель, содержащая генератор и дискриминаторы.
            batch: словарь с ключами:
                "audio": [B, 1, T] - референс звук
                "pred_audio": [B, 1, T'] - предсказанный звук (уже посчитан моделью)

        Returns:
            dict:
                {
                    "loss_gen": ...,
                    "loss_disc": ...,
                    "loss_mel": ...,
                    "loss_fm": ...,
                    "loss_adv_gen": ...
                }
        """
        real_audio = batch["audio"]  # [B, 1, T]
        pred_audio = batch["pred_audio"]  # [B, 1, T']

        # Извлечём мел-спектрограммы
        with torch.no_grad():
            mel_real = self.mel_extractor(real_audio.squeeze(1))
        mel_pred = self.mel_extractor(pred_audio.squeeze(1))

        #TODO посмотреть на размеры спек
        # Обрезка (если требуется) до одинаковой длины:
        min_len = min(mel_real.shape[-1], mel_pred.shape[-1])
        mel_real = mel_real[..., :min_len]
        mel_pred = mel_pred[..., :min_len]

        # Mel reconstruction loss
        mel_loss = self.l1(mel_pred, mel_real)

        # Дискриминаторский лосс:
        # Для дискриминатора: мы хотим отличить real от fake
        disc_real = model.discriminate(real_audio)  # list of (out, feats)
        disc_fake = model.discriminate(pred_audio.detach())  # detach важен для дискриминатора

        disc_loss = 0.0
        for (f_out, _), (r_out, _) in zip(disc_fake, disc_real):
            # На реальном аудио дискриминатору хотим: out ~ 1
            disc_loss += F.mse_loss(r_out, torch.ones_like(r_out))
            # На фейковом аудио: out ~ 0
            disc_loss += F.mse_loss(f_out, torch.zeros_like(f_out))
        disc_loss /= len(disc_fake)

        # Генераторский лосс:
        # Генератору нужен adversarial лосс (хочет f_out ~ 1), feature matching loss и mel_loss
        disc_fake_g = model.discriminate(pred_audio)  # без detach для градиентов в генератор
        adv_loss_gen = 0.0
        fm_loss = 0.0

        for (f_out, f_feats), (r_out, r_feats) in zip(disc_fake_g, disc_real):
            adv_loss_gen += F.mse_loss(f_out, torch.ones_like(f_out))
            # Feature Matching: сравниваем промежуточные фичи
            for ff, rf in zip(f_feats, r_feats):
                # TODO fix demensions
                min_len = min(ff.shape[-1], rf.shape[-1])
                ff = ff[..., :min_len]
                rf = rf[..., :min_len]

                fm_loss += F.l1_loss(ff, rf)

        adv_loss_gen /= len(disc_fake_g)
        fm_loss /= len(disc_fake_g)

        loss_gen = adv_loss_gen + 2.0 * fm_loss + 45.0 * mel_loss

        return {
            "loss_gen": loss_gen,
            "loss_disc": disc_loss,
            "loss_mel": mel_loss,
            "loss_fm": fm_loss,
            "loss_adv_gen": adv_loss_gen,
        }
