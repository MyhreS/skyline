import scipy
import librosa
import numpy as np
import random


class Augmenter:
    augment_options = ["low_pass", "pitch_shift", "add_noise", "high_pass", "band_pass"]

    def __init__(self):
        pass

    def augment(self, wav, sample_rate, augmentation: str):
        if augmentation == "low_pass":
            wav = self.apply_low_pass_filter(wav, sample_rate)
        elif augmentation == "pitch_shift":
            n_steps = random.uniform(-2, 2)
            wav = self.apply_pitch_shift(wav, sample_rate, n_steps)
        elif augmentation == "add_noise":
            wav = self.apply_noise(wav)
        elif augmentation == "high_pass":
            wav = self.apply_high_pass_filter(wav, sample_rate)
        elif augmentation == "band_pass":
            wav = self.apply_band_pass_filter(wav, sample_rate)
        return wav

    def apply_low_pass_filter(self, wav, sample_rate, cutoff_freq=2000):
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = scipy.signal.butter(4, normal_cutoff, btype="low", analog=False)
        filtered_wav = scipy.signal.lfilter(b, a, wav)
        return filtered_wav

    def apply_pitch_shift(self, wav, sample_rate, n_steps=0):
        return librosa.effects.pitch_shift(y=wav, sr=sample_rate, n_steps=n_steps)

    def apply_noise(self, wav, noise_level=0.005):
        noise = np.random.randn(len(wav))
        augmented_wav = wav + noise_level * noise
        return np.clip(augmented_wav, -1.0, 1.0)

    def apply_high_pass_filter(self, wav, sample_rate, cutoff_freq=2000):
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = scipy.signal.butter(4, normal_cutoff, btype="high", analog=False)
        filtered_wav = scipy.signal.lfilter(b, a, wav)
        return filtered_wav

    def apply_band_pass_filter(
        self, wav, sample_rate, low_cutoff=500, high_cutoff=3000
    ):
        nyquist = 0.5 * sample_rate
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        b, a = scipy.signal.butter(4, [low, high], btype="band", analog=False)
        filtered_wav = scipy.signal.lfilter(b, a, wav)
        return filtered_wav
