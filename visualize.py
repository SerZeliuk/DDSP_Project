import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def plot_waveform(ax, audio, sr, title="Waveform"):
    times = np.arange(len(audio)) / sr
    ax.plot(times, audio)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")


def plot_f0_confidence(ax, f0, confidence, frame_rate):
    times = np.arange(len(f0)) / frame_rate
    ax.plot(times, f0, label="f0 (Hz)")
    ax.set_ylabel("Frequency (Hz)")
    ax_twin = ax.twinx()
    ax_twin.plot(times, confidence, color="gray", alpha=0.5, label="Confidence")
    ax_twin.set_ylabel("Confidence")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper left")
    ax_twin.legend(loc="upper right")
    ax.set_title("Pitch & Confidence")


def plot_loudness(ax, loudness, frame_rate):
    times = np.arange(len(loudness)) / frame_rate
    ax.plot(times, loudness)
    ax.set_title("Loudness (dB)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dB")


def plot_spectrogram(ax, audio, sr, hop_length, title="Spectrogram"):
    # Compute magnitude spectrogram in dB
    S = np.abs(librosa.stft(audio, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db,
                                   sr=sr,
                                   hop_length=hop_length,
                                   x_axis='time',
                                   y_axis='hz',
                                   ax=ax)
    ax.set_title(title)
    plt.colorbar(img, ax=ax, format='%+2.0f dB')


def plot_input(audio, sr, f0, confidence, loudness, hop_length, frame_rate):
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), constrained_layout=True)
    plot_waveform(axs[0], audio, sr, title="Input Waveform")
    plot_f0_confidence(axs[1], f0, confidence, frame_rate)
    plot_loudness(axs[2], loudness, frame_rate)
    plot_spectrogram(axs[3], audio, sr, hop_length, title="Input Spectrogram")
    plt.show()


def plot_output(audio, sr, f0, loudness, hop_length, frame_rate):
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), constrained_layout=True)
    plot_waveform(axs[0], audio, sr, title="Output Waveform")
    axs[1].plot(np.arange(len(f0)) / frame_rate, f0)
    axs[1].set_title("Output f0 (Hz)")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Frequency (Hz)")
    plot_loudness(axs[2], loudness, frame_rate)
    plot_spectrogram(axs[3], audio, sr, hop_length, title="Output Spectrogram")
    plt.show()
