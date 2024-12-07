import torch
import torch.nn as nn

from dataclasses import dataclass
import torchaudio
import librosa

######################################################
## HiFi-GAN Generator
## См. статью: генератор использует несколько уровней апсэмплинга
## и наборы резидуальных блоков (ResBlock1, ResBlock2).
######################################################

class ResBlock(nn.Module):
    """
    ResBlock из HiFi-GAN (упрощённая версия Type1/Type2).
    Используем список задаваемых расширений (dilations) и ядер (kernels).
    """
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilations:
            padding = ((kernel_size - 1) * d) // 2
            self.convs.append(
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(channels, channels, kernel_size, 1, padding=padding, dilation=d),
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(channels, channels, kernel_size, 1, padding=padding, dilation=d)
                )
            )

    def forward(self, x):
        for conv in self.convs:
            residual = x
            out = conv(x)
            x = out + residual
        return x


class HiFiGANGenerator(nn.Module):
    """
    Генератор HiFi-GAN.
    Параметры:
    - in_channels: количество мел-банок (обычно 80)
    - upsample_rates: список апсэмплинг факторов
    - upsample_initial_channel: начальное число каналов в ConvTranspose1d
    - resblock_kernel_sizes, resblock_dilation_sizes: параметры ресблоков
    """
    def __init__(
        self,
        in_channels=80,
        upsample_rates=[8,8,2,2],
        upsample_initial_channel=512,
        resblock_kernel_sizes=[3,7,11],
        resblock_dilation_sizes=[[1,3,5],[1,3,5],[1,3,5]]
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)

        self.input_conv = nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)

        self.ups = nn.ModuleList()
        in_ch = upsample_initial_channel
        for u in upsample_rates:
            self.ups.append(
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.utils.weight_norm(
                        nn.ConvTranspose1d(in_ch, in_ch//2, u*2, u, (u//2 + u%2))
                    )
                )
            )
            in_ch = in_ch // 2

        # Для каждого апсэмплинга набор резблоков
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = in_ch * (2**(-i)) # или просто in_ch после апсэмплинга, но упрощаем
            # фактически ch обновляется после каждого апсэмпла
            # после апсэмпла in_ch уменьшился вдвое
            # поэтому достать правильно текущий ch можно иначе:
            # после всех апсэмплов in_ch = upsample_initial_channel // (2**len(upsample_rates))
            # Для упрощения: текущий канал после i-го апсэмпла:
            current_ch = upsample_initial_channel // (2**(i+1))

            for (k, d) in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(current_ch, k, d))

        self.out_conv = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(current_ch, 1, 7, 1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input_conv(x)
        rb_count = self.num_kernels * len(self.ups)
        idx = 0
        for i, up in enumerate(self.ups):
            x = up(x)
            # Проходим по num_kernels ресблоков для этого апсэмплинга и усредняем результаты
            sum_rb = 0
            for _ in range(self.num_kernels):
                sum_rb += self.resblocks[idx](x)
                idx += 1
            x = sum_rb / self.num_kernels

        x = self.out_conv(x)
        return x


######################################################
## HiFi-GAN Multi-Scale Discriminators
## В статье HiFi-GAN используются Multi-Scale и Multi-Period дискриминаторы.
## Здесь приведём Multi-Scale из статьи:
## Три дискриминатора на разных временных масштабах.
######################################################

class HiFiGANDiscriminatorScale(nn.Module):
    """
    Один дискриминатор для определённой шкалы.
    """
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(1, 128, 15, 1, 7)),
            nn.utils.weight_norm(nn.Conv1d(128, 128, 41, 4, 20, groups=4)),
            nn.utils.weight_norm(nn.Conv1d(128, 256, 41, 4, 20, groups=16)),
            nn.utils.weight_norm(nn.Conv1d(256, 512, 41, 4, 20, groups=16)),
            nn.utils.weight_norm(nn.Conv1d(512, 1024, 41, 4, 20, groups=16)),
            nn.utils.weight_norm(nn.Conv1d(1024, 1024, 5, 1, 2)),
            nn.utils.weight_norm(nn.Conv1d(1024, 1, 3, 1, 1))
        ])
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        feats = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i < len(self.model)-1:
                x = self.leaky_relu(x)
            feats.append(x)
        return feats[-1], feats[:-1]

class HiFiGANMultiScaleDiscriminator(nn.Module):
    def __init__(self, num_scales=3, scale_factor=2):
        super().__init__()
        self.discriminators = nn.ModuleList()
        for i in range(num_scales):
            self.discriminators.append(HiFiGANDiscriminatorScale())
        self.pooling = nn.AvgPool1d(kernel_size=scale_factor, stride=scale_factor, padding=0)

    def forward(self, x):
        # x: [B, 1, T]
        outs = []
        for i, d in enumerate(self.discriminators):
            out, feats = d(x)
            outs.append((out, feats))
            if i < len(self.discriminators)-1:
                x = self.pooling(x)
        return outs


######################################################
## Полная модель HiFi-GAN (Generator + Multi-Scale Discriminator)
## Можно добавить Multi-Period Discriminator по аналогии.
######################################################

class HiFiGAN(nn.Module):
    def __init__(self, generator_params, discriminator_params):
        super().__init__()
        self.generator = HiFiGANGenerator(**generator_params)
        self.msd = HiFiGANMultiScaleDiscriminator(**discriminator_params)

    def forward(self, mel, **batch):
        """
        batch ожидается с ключами:
        "mel": [B, n_mels, T_mel]
        "audio": [B, 1, T] - ground truth
        Возвращаем:
        batch с "pred_audio": [B, 1, T']
        """
        pred_audio = self.generator(mel)
        batch["pred_audio"] = pred_audio
        batch["model"] = self
        return batch

    def discriminate(self, audio):
        # Прогоняем аудио через мультискейл дискриминатор
        return self.msd(audio)
