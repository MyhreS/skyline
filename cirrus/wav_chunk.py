
import os
import logging
import librosa
import numpy as np
import scipy
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')


class WavChunk():
    def __init__(self, wav_blob, label, relative_start_sec, relative_end_sec, duration_sec):
        self.wav_blob = wav_blob
        self.wav_id = wav_blob.split('/')[-1].split('.')[0]
        self.label = label
        self.relative_start_sec = relative_start_sec
        self.relative_end_sec = relative_end_sec
        self.duration_sec = duration_sec
        self.class_ = None
        self.split = None
        self.augmentation = None

    def set_class(self, class_):
        self.class_ = class_

    def set_split(self, split):
        self.split = split
    
    def add_augmentation(self, augmentation):
        if self.split != 'train':
            raise ValueError("Can only add augmentation to training wavs")
        list_of_augmentations = ['low_pass', 'pitch_shift', 'add_noise', 'high_pass', 'band_pass']
        if augmentation not in list_of_augmentations:
            raise ValueError("Invalid augmentation: %s. Choose from %s", augmentation, list_of_augmentations)
        self.augmentation = augmentation
    
    def copy(self):
        new_wav_chunk = WavChunk(
            self.wav_blob,
            self.label,
            self.relative_start_sec,
            self.relative_end_sec,
            self.duration_sec
        )
        if self.class_ is not None:
            new_wav_chunk.set_class(self.class_)
        if self.split is not None:
            new_wav_chunk.set_split(self.split)
        if self.augmentation is not None:
            new_wav_chunk.add_augmentation(self.augmentation)
        return new_wav_chunk

    
    def write(self, input_path_to_data, output_path_to_data, audio_format, file_type):
        if self.split is None:
            raise ValueError("Wav chunk does not have a split")
        if self.class_ is None:
            raise ValueError("Wav chunk does not have a class")
        
        if file_type != "npy":
            raise ValueError("Only npy file type is supported")
        
        file_hash = self._get_file_hash(audio_format)
        # Check if file already exists
        if os.path.exists(os.path.join(output_path_to_data, self.split, self.class_, file_hash + "." + file_type)):
            logging.info("File %s already exists", file_hash)
            return

        wav, sample_rate = librosa.load(os.path.join(input_path_to_data, self.wav_blob), sr=44100)
        wav_chunk = wav[int(self.relative_start_sec * sample_rate):int(self.relative_end_sec * sample_rate)]

        if self.augmentation is not None:
            if self.augmentation == 'low_pass':
                wav_chunk = self._apply_low_pass_filter(wav_chunk, sample_rate)
            elif self.augmentation == 'pitch_shift':
                n_steps = random.uniform(-2, 2)
                wav_chunk = self._apply_pitch_shift(wav_chunk, sample_rate, n_steps)
            elif self.augmentation == 'add_noise':
                wav_chunk = self._apply_noise(wav_chunk)
            elif self.augmentation == 'high_pass':
                wav_chunk = self._apply_high_pass_filter(wav_chunk, sample_rate)
            elif self.augmentation == 'band_pass':
                wav_chunk = self._apply_band_pass_filter(wav_chunk, sample_rate)
            else:
                raise ValueError("Invalid augmentation: %s", self.augmentation)
        
        spectrogram = self._to_spectrogram(wav_chunk, sample_rate, audio_format)
        self._save(spectrogram, output_path_to_data, file_hash, file_type)

    def _get_file_hash(self, audio_format):
        augmentation = "none" if self.augmentation is None else self.augmentation
        return f"{self.wav_id}_{int(self.relative_start_sec)}_{int(self.relative_end_sec)}_{augmentation}_{audio_format}"

    def _apply_low_pass_filter(self, wav, sample_rate, cutoff_freq=2000):
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = scipy.signal.butter(4, normal_cutoff, btype='low', analog=False)
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
        b, a = scipy.signal.butter(4, normal_cutoff, btype='high', analog=False)
        filtered_wav = scipy.signal.lfilter(b, a, wav)
        return filtered_wav
    
    def _apply_band_pass_filter(self, wav, sample_rate, low_cutoff=500, high_cutoff=3000):
        nyquist = 0.5 * sample_rate
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        b, a = scipy.signal.butter(4, [low, high], btype='band', analog=False)
        filtered_wav = scipy.signal.lfilter(b, a, wav)
        return filtered_wav
    
    def _to_spectrogram(self, wav, sample_rate, spectrogram_type):
        if spectrogram_type == 'logmel':
            mel_spectrogram = librosa.feature.melspectrogram(y=wav, sr=sample_rate, n_mels=128, n_fft=2048, hop_length=512, fmin=0, fmax=44100 // 2)
            spectrogram = librosa.power_to_db(mel_spectrogram)
            return spectrogram
        elif spectrogram_type == 'stft':
            stft = librosa.stft(wav, n_fft=2048, hop_length=512)
            spectrogram = np.abs(stft)
            return spectrogram
        else:
            raise ValueError("Invalid spectrogram_type. Choose 'logmel' or 'stft'.")

    def _save(self, spectrogram, output_path_to_data, file_hash, file_type):
        # Save as npy
        save_dir_path = os.path.join(output_path_to_data, self.split, self.class_)
        npy_filename = os.path.join(save_dir_path, file_hash + "." + file_type)
        logging.info("Writing %s", npy_filename)
        np.save(npy_filename, spectrogram)