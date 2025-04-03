import torch.nn as nn
import torch

from src.utils.mel import MelSpectrogram

class ResBlock(nn.Module):
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


        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
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
        idx = 0
        for i, up in enumerate(self.ups):
            x = up(x)
            sum_rb = 0
            for _ in range(self.num_kernels):
                sum_rb += self.resblocks[idx](x)
                idx += 1
            x = sum_rb / self.num_kernels

        x = self.out_conv(x)
        return x


######################################################
## HiFi-GAN Multi-Scale Discriminators
######################################################

class HiFiGANDiscriminatorScale(nn.Module):
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

class HiFiGAN(nn.Module):
    def __init__(self, generator_params, discriminator_params, mel_config=None):
        super().__init__()
        self.generator = HiFiGANGenerator(**generator_params)
        self.msd = HiFiGANMultiScaleDiscriminator(**discriminator_params)

        self.mel_config = mel_config
        if self.mel_config:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.mel_extractor = MelSpectrogram(self.mel_config, device=device)

    def forward(self, **batch):
        if self.mel_config and "mel" not in batch:
            batch["mel"] = self.mel_extractor(batch["audio"]).squeeze(1)
        pred_audio = self.generator(batch["mel"])
        if "audio" in batch:
            if pred_audio[0].shape[-1] > batch["audio"].shape[-1]:
                pred_audio = pred_audio[..., :batch["audio"].shape[-1]]
            elif pred_audio[0].shape[-1] < batch["audio"].shape[-1]:
                raise ValueError("Predicted audio is shorter than original audio")
        batch["pred_audio"] = pred_audio
        return batch

    def discriminate(self, audio):
        return self.msd(audio)

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        generator_parameters = sum([p.numel() for p in self.generator.parameters()])
        discriminator_parameters = sum([p.numel() for p in self.msd.parameters()])

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"
        result_info = result_info + f"\nGenerator parameters: {generator_parameters}"
        result_info = result_info + f"\nDiscriminator parameters: {discriminator_parameters}"

        return result_info
