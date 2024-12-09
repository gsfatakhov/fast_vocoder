import torchaudio
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_audio(file_path, target_sr=22050):
    waveform, sr = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = waveform[0].unsqueeze(0)
    if sr != target_sr:
        print(f"Resampling from {sr} to {target_sr} for {file_path}")
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
    return waveform.squeeze(0), target_sr


def plot_spectrogram(ax, waveform, sr, title):
    spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=80)(waveform)
    log_spectrogram = torchaudio.transforms.AmplitudeToDB()(spec)
    ax.imshow(log_spectrogram.squeeze(0).numpy(), aspect='auto', origin='lower')
    ax.set_title(title)
    ax.set_xlabel("Время (Frames)")
    ax.set_ylabel("Частота (Mels)")


def plot_waveform(ax, waveform, title):
    ax.plot(waveform.numpy())
    ax.set_title(title)
    ax.set_xlabel("Время (Samples)")
    ax.set_ylabel("Амплитуда")


def compare_audio(gt_path, gen_path):
    gt_files = {file.name: file for file in Path(gt_path).glob("*.wav")}
    gen_files = {file.name: file for file in Path(gen_path).glob("*.wav")}

    common_files = set(gt_files.keys()).intersection(gen_files.keys())
    if not common_files:
        print("Нет совпадающих файлов для сравнения.")
        print(f"GT: {gt_files.keys()}")
        print(f"Generated: {gen_files.keys()}")
        return

    for file_name in common_files:
        gt_waveform, gt_sr = load_audio(gt_files[file_name])
        gen_waveform, gen_sr = load_audio(gen_files[file_name])

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Сравнение файла: {file_name}", fontsize=16)

        plot_waveform(axs[0, 0], gt_waveform, "GT Waveform")
        plot_waveform(axs[0, 1], gen_waveform, "Generated Waveform")
        plot_spectrogram(axs[1, 0], gt_waveform, gt_sr, "GT Spectrogram")
        plot_spectrogram(axs[1, 1], gen_waveform, gen_sr, "Generated Spectrogram")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сравнение аудиофайлов.")
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--gen_path", type=str, required=True)
    args = parser.parse_args()

    compare_audio(args.gt_path, args.gen_path)
