import os
import scipy
import librosa
import numpy as np
import random
import pandas as pd
from typing import List


def normalize_audio_energy(audio, target_energy=1.0):
    current_energy = np.sum(np.square(audio))
    normalization_factor = np.sqrt(target_energy / (current_energy + 1e-10))
    normalized_audio = audio * normalization_factor
    return normalized_audio


class Augmenter:
    augment_options = [
        "low_pass",
        "pitch_shift",
        "add_noise",
        "high_pass",
        "band_pass",
        "mix_1",
        "mix_2",
        "mix_3",
        "mix_4",
        "mix_5",
    ]

    def __init__(self, path_to_input_data: str):
        path_to_mix_data = path_to_input_data.split("/")[0:-1] + ["mixdata/wavs"]
        self.path_to_mix_data = "/".join(path_to_mix_data)
        self.data_in_mixdata = os.listdir(self.path_to_mix_data)
        if len(self.data_in_mixdata) == 0:
            raise ValueError("No files found in the mixdata folder")

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
            wav = self._apply_low_pass_filter(wav, sample_rate)
        elif augmentation == "pitch_shift":
            n_steps = random.uniform(-2, 2)
            wav = self._apply_pitch_shift(wav, sample_rate, n_steps)
        elif augmentation == "add_noise":
            wav = self._apply_noise(wav)
        elif augmentation == "high_pass":
            wav = self._apply_high_pass_filter(wav, sample_rate)
        elif augmentation == "band_pass":  # Might cut off all drone sounds
            wav = self._apply_band_pass_filter(wav, sample_rate)
        elif augmentation.startswith("mix"):
            wav = self._apply_mix(wav, sample_rate)
        return wav

    def _apply_low_pass_filter(self, wav, sample_rate, min_cutoff=500, max_cutoff=7500):
        # Generate a random cutoff frequency between min_cutoff and max_cutoff
        cutoff_freq = random.randint(min_cutoff, max_cutoff)

        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = scipy.signal.butter(4, normal_cutoff, btype="low", analog=False)
        filtered_wav = scipy.signal.lfilter(b, a, wav)

        # Assuming normalize_audio_energy is a function defined elsewhere that normalizes the audio energy
        return normalize_audio_energy(filtered_wav)

    def _apply_pitch_shift(self, wav, sample_rate, n_steps=5000):
        random_n_steps = np.random.uniform(-n_steps, n_steps)
        return normalize_audio_energy(
            librosa.effects.pitch_shift(y=wav, sr=sample_rate, n_steps=random_n_steps)
        )

    def _apply_noise(self, wav, max_noise_level=0.0005):
        random_noise_level = np.random.uniform(0.00005, max_noise_level)
        noise = np.random.randn(len(wav))
        augmented_wav = wav + random_noise_level * noise
        return normalize_audio_energy(np.clip(augmented_wav, -1.0, 1.0))

    def _apply_high_pass_filter(
        self, wav, sample_rate, min_cutoff=500, max_cutoff=4500
    ):
        cutoff_freq = random.randint(min_cutoff, max_cutoff)
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = scipy.signal.butter(4, normal_cutoff, btype="high", analog=False)
        filtered_wav = scipy.signal.lfilter(b, a, wav)
        return normalize_audio_energy(filtered_wav)

    def _apply_band_pass_filter(
        self,
        wav,
        sample_rate,
        low_min_cutoff=500,
        low_max_cutoff=3500,
        high_min_cutoff=4500,
        high_max_cutoff=7500,
    ):

        low_cutoff_freq = random.randint(low_min_cutoff, low_max_cutoff)
        high_cutoff = random.randint(high_min_cutoff, high_max_cutoff)

        nyquist = 0.5 * sample_rate
        low = low_cutoff_freq / nyquist
        high = high_cutoff / nyquist
        b, a = scipy.signal.butter(4, [low, high], btype="band", analog=False)
        filtered_wav = scipy.signal.lfilter(b, a, wav)
        return normalize_audio_energy(filtered_wav)

    def _apply_mix(
        self,
        wav_chunk,
        sample_rate: int = 16000,
        mix_ratio_from: float = 0.99,
        mix_ratio_to: float = 0.4,
    ):
        """
        Mixes wav_chunk with a randomly selected audio chunk.

        Args:
        - wav_chunk: The primary audio chunk to mix.
        - sample_rate: The sample rate of the audio chunks.
        - mix_ratio: The ratio of the secondary chunk to mix with the primary chunk.

        Returns:
        - The mixed audio chunk.
        """

        secondary_chunk = self._read_random_mix_file(
            chunk_size=len(wav_chunk), sample_rate=sample_rate
        )
        assert len(secondary_chunk) == len(
            wav_chunk
        ), "The secondary chunk must have the same length as the wav_chunk"
        mix_ratio = np.random.uniform(mix_ratio_from, mix_ratio_to)

        assert 0 <= mix_ratio <= 1, "The mixing ratio must be between 0 and 1"

        # Mixing the chunks
        mixed_chunk = wav_chunk + (secondary_chunk * mix_ratio)
        return normalize_audio_energy(mixed_chunk)

    def _read_random_mix_file(self, chunk_size: int, sample_rate: int = 16000):
        """
        Takes a random audio file from the mixdata folder and returns a chunk of the audio file equal to the chunk_size.

        Args:
        - chunk_size: The length of the audio chunk to return.
        - sample_rate: The sample rate of the audio chunks.

        Returns:
        - The audio chunk.
        """

        file_loaded = False
        mix_wav = None

        while not file_loaded:
            try:
                random_file = np.random.choice(self.data_in_mixdata)
                mix_wav, sr = librosa.load(
                    self.path_to_mix_data + "/" + random_file, sr=sample_rate
                )
                if chunk_size > len(mix_wav):
                    continue
                file_loaded = True

            except Exception as e:
                print(f"Error loading file: {e}")
                continue

        # Once a valid file is loaded, extract the chunk
        start = np.random.randint(0, len(mix_wav) - chunk_size)
        end = start + chunk_size
        return normalize_audio_energy(mix_wav[start:end])
