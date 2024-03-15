import scipy
import librosa
import numpy as np
import random
import pandas as pd
from typing import List


class Augmenter:
    augment_options = ["low_pass", "pitch_shift", "add_noise", "high_pass", "band_pass"]

    def augment_df_files(
        self, df: pd.DataFrame, augmentations: List, only_augment_drone: bool
    ):
        assert len(df) > 0, "Dataframe is empty"
        assert "split" in df.columns, "Dataframe does not contain split column"
        df["augmentation"] = None
        if augmentations is None:
            return df

        # Get a non_test_df which does not contain a split containing the word "test"
        non_test_df = df[~df["split"].str.contains("test")]

        type_of_drones = [
            "electric_quad_drone",
            "racing_drone",
            "electric_fixedwing_drone",
            "petrol_fixedwing_drone",
        ]
        if only_augment_drone:
            non_test_df = non_test_df[non_test_df["label"].isin(type_of_drones)]

        # Create a DataFrame for each augmentation
        augmented_dfs = [non_test_df.copy() for _ in augmentations]
        # Assign the augmentation labels
        for aug_df, augmentation in zip(augmented_dfs, augmentations):
            aug_df["augmentation"] = augmentation
        # Concatenate all augmented DataFrames along with the original non-test DataFrame
        result_df = pd.concat(augmented_dfs + [df], ignore_index=True)
        return result_df

    def augment_file(self, wav: np.ndarray, sample_rate: int, augmentation: str):
        if augmentation == "low_pass":  # Might cut off all drone sounds
            wav = self._apply_low_pass_filter(wav, sample_rate, cutoff_freq=5000)
        elif augmentation == "pitch_shift":
            n_steps = random.uniform(-2, 2)
            wav = self._apply_pitch_shift(wav, sample_rate, n_steps)
        elif augmentation == "add_noise":
            wav = self._apply_noise(wav)
        elif augmentation == "high_pass":
            wav = self._apply_high_pass_filter(wav, sample_rate)
        elif augmentation == "band_pass":  # Might cut off all drone sounds
            wav = self._apply_band_pass_filter(wav, sample_rate, high_cutoff=5000)
        return wav

    def _apply_low_pass_filter(self, wav, sample_rate, cutoff_freq=2000):
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = scipy.signal.butter(4, normal_cutoff, btype="low", analog=False)
        filtered_wav = scipy.signal.lfilter(b, a, wav)
        return filtered_wav

    def _apply_pitch_shift(self, wav, sample_rate, n_steps=0):
        return librosa.effects.pitch_shift(y=wav, sr=sample_rate, n_steps=n_steps)

    def _apply_noise(self, wav, noise_level=0.005):
        noise = np.random.randn(len(wav))
        augmented_wav = wav + noise_level * noise
        return np.clip(augmented_wav, -1.0, 1.0)

    def _apply_high_pass_filter(self, wav, sample_rate, cutoff_freq=2000):
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = scipy.signal.butter(4, normal_cutoff, btype="high", analog=False)
        filtered_wav = scipy.signal.lfilter(b, a, wav)
        return filtered_wav

    def _apply_band_pass_filter(
        self, wav, sample_rate, low_cutoff=500, high_cutoff=3000
    ):
        nyquist = 0.5 * sample_rate
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        b, a = scipy.signal.butter(4, [low, high], btype="band", analog=False)
        filtered_wav = scipy.signal.lfilter(b, a, wav)
        return filtered_wav
