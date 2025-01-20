import torch
import torch.nn as nn

from src.model.src_lightvoc.conformer import ConformerBlock

class LightVocGenerator(nn.Module):
    def __init__(
            self,
            n_mels=80,
            d_model=256,
            d_ff=1024,
            n_heads=4,
            num_conformer_blocks=6,
            conv_kernel_size=31,
            n_fft=1024,
            hop_length=256,
            win_length=256,
            dropout=0.1
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # Первый сверточный слой для приведения мел-спектрограммы в нужное пространство
        self.input_conv = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1)

        # Стек Conformer-блоков
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model, d_ff, n_heads, conv_kernel_size, dropout)
            for _ in range(num_conformer_blocks)
        ])

        # Финальный сверточный слой для получения коэффициентов STFT (реальная и мнимая части)
        # Выходное число каналов = (n_fft // 2 + 1) * 2
        self.output_conv = nn.Conv1d(d_model, (n_fft // 2 + 1) * 2, kernel_size=1)

    def forward(self, mel):
        """
        mel: [B, n_mels, T]
        """
        x = self.input_conv(mel)  # [B, d_model, T]
        # Перестроение для обработки Conformer-блоками: [B, T, d_model]
        x = x.transpose(1, 2)

        for block in self.conformer_blocks:
            x = block(x)

        # Возвращаемся к форме [B, d_model, T] для свёрточного слоя
        x = x.transpose(1, 2)
        stft_out = self.output_conv(x)  # [B, (n_fft//2+1)*2, T]

        B, channels, T = stft_out.shape
        stft_out = stft_out.view(B, 2, self.n_fft // 2 + 1, T)
        real = stft_out[:, 0, :, :]
        imag = stft_out[:, 1, :, :]
        complex_stft = torch.complex(real, imag)

        # Обратное преобразование STFT (iSTFT) для получения waveform
        waveform = torch.istft(
            complex_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            length=None
        )
        return waveform
