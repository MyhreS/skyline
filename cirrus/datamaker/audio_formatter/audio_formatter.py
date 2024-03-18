import pandas as pd
import librosa
import numpy as np


class AudioFormatter:
    audio_format_options = ["stft", "log_mel", "waveform"]

    def audio_format_df_files(self, df: pd.DataFrame, audio_format: str):
        assert len(df) > 0, "df must be non-empty"
        df["audio_format"] = audio_format
        return df

    def audio_format_file(self, wav: np.ndarray, sample_rate: int, audio_format: str):
        if audio_format == "log_mel":
            return self.to_log_mel_spectrogram(wav, sample_rate)
        elif audio_format == "stft":
            return self.to_stft_spectrogram(wav, sample_rate)
        elif audio_format == "waveform":
            return wav
        else:
            raise ValueError(f"Audio format {audio_format} not supported")

    def to_log_mel_spectrogram(self, wav: np.ndarray, sample_rate: int):
        mel_spectrogram = librosa.feature.melspectrogram(
            y=wav,
            sr=sample_rate,
            n_mels=512,
            n_fft=4096,
            hop_length=512,
            fmin=100,
            fmax=6500,
        )
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        return log_mel_spectrogram

    def to_stft_spectrogram(self, wav: np.ndarray, sample_rate: int):
        stft = librosa.stft(wav, n_fft=4096, hop_length=512)
        stft_spectrogram = np.abs(stft)
        return stft_spectrogram
