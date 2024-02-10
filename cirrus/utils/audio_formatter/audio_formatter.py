import pandas as pd
import librosa
import numpy as np


class AudioFormatter:
    audio_format_options = ["stft", "log_mel"]

    def audio_format_df_files(self, df: pd.DataFrame, audio_format: str):
        assert len(df) > 0, "df must be non-empty"
        df["audio_format"] = audio_format
        return df

    def to_spectrogram(self, wav: np.ndarray, sample_rate: int, spectrogram_type: str):
        spectrogram = None
        if spectrogram_type == "logmel":
            mel_spectrogram = librosa.feature.melspectrogram(
                y=wav,
                sr=sample_rate,
                n_mels=128,
                n_fft=2048,
                hop_length=512,
                fmin=0,
                fmax=sample_rate // 2,
            )
            spectrogram = librosa.power_to_db(mel_spectrogram)
        elif spectrogram_type == "stft":
            stft = librosa.stft(wav, n_fft=2048, hop_length=512)
            spectrogram = np.abs(stft)
        else:
            raise ValueError("Invalid spectrogram_type. Choose 'logmel' or 'stft'.")
        return spectrogram
