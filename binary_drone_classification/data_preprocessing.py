

import os
import shutil
from dotenv import load_dotenv
import logging
import pandas as pd
import librosa
import numpy as np
import librosa
import numpy as np
import os
import scipy
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
load_dotenv()


class Data():
    def __init__(self, input_path_to_data):
        logging.info("Setting up data class")
        self.input_path_to_data = input_path_to_data
        logging.info("Set up path to data: %s", self.input_path_to_data)
        self.metadata_df = pd.read_csv(os.path.join(input_path_to_data, "data.csv"))
        logging.info("Found metadata csv with shape: %s", self.metadata_df.shape)
        self.output_path_to_data = "cache/data"
        self.wavs = {}

    def load_data(self, window_size = 2.0):
        logging.info("Loading data")
        wavs = {}
        for _, row in self.metadata_df.iterrows():
            wav = WavChunk(row['wav_blob'], row['label'], row['relative_start_sec'], row['relative_end_sec'], row['duration_sec'])
            if wav.wav_id in wavs:
                wavs[wav.wav_id].append(wav)
            else:
                wavs[wav.wav_id] = [wav]

        # Concatinate wav chunks on with same label
        for wav_id, wav_chunks in wavs.items():
            wav_chunks.sort(key=lambda x: x.relative_start_sec)

            new_wav_chunks = []
            for i in range(len(wav_chunks)):
                if i == 0:
                    new_wav_chunks.append(wav_chunks[i])
                else:
                    new_wav_chunk = self._concatenate_two_wav_chunks(new_wav_chunks[-1], wav_chunks[i])
                    if new_wav_chunk is None:
                        new_wav_chunks.append(wav_chunks[i])
                    else:
                        new_wav_chunks[-1] = new_wav_chunk
            wavs[wav_id] = new_wav_chunks

        # Window wav chunks
        for wav_id, wav_chunks in wavs.items():
            new_wav_chunks = []
            for wav_chunk in wav_chunks:
                new_wav_chunks.extend(self._window_wav_chunk(wav_chunk, window_size))
            wavs[wav_id] = new_wav_chunks

        # Check that all wav chunks are of the same length
        for wav_id, wav_chunks in wavs.items():
            for wav_chunk in wav_chunks:
                if wav_chunk.duration_sec != window_size:
                    raise ValueError("Not all wav chunks are of the same length")

        self.wavs = wavs
        unique_wav_ids = len(wavs)
        number_of_wav_chunks = sum([len(wav_chunks) for wav_chunks in wavs.values()])
        logging.info("Loaded %d wav chunks with %d unique wav id's", number_of_wav_chunks, unique_wav_ids)

        # Print the different labels and their counts
        labels = [wav_chunk.label for wav_chunks in wavs.values() for wav_chunk in wav_chunks]
        unique_labels = set(labels)
        for label in unique_labels:
            logging.info("Found that label %s has %d wav chunks", label, labels.count(label))

    def _concatenate_two_wav_chunks(self, wav_chunk_a, wav_chunk_b):
        if wav_chunk_a.wav_id == wav_chunk_b.wav_id:
            if wav_chunk_a.label == wav_chunk_b.label:
                if wav_chunk_a.relative_end_sec == wav_chunk_b.relative_start_sec:
                    new_wav_chunk = WavChunk(
                        wav_chunk_a.wav_blob,
                        wav_chunk_a.label,
                        wav_chunk_a.relative_start_sec,
                        wav_chunk_b.relative_end_sec,
                        wav_chunk_a.duration_sec + wav_chunk_b.duration_sec
                    )
                    return new_wav_chunk
        return None
    
    def _window_wav_chunk(self, wav_chunk, window_size):
        new_wav_chunks = []
        window_start_sec = wav_chunk.relative_start_sec
        window_end_sec = wav_chunk.relative_start_sec + window_size
        while True:
            if window_end_sec > wav_chunk.relative_end_sec:
                break
            new_wav_chunk = WavChunk(
                wav_chunk.wav_blob,
                wav_chunk.label,
                window_start_sec,
                window_end_sec,
                window_size
            )
            new_wav_chunks.append(new_wav_chunk)
            window_start_sec += window_size
            window_end_sec += window_size
        return new_wav_chunks

    def map_labels_to_classes(self, label_to_class_map):
        logging.info("Mapping labels to classes")
        for _, wav_chunks in self.wavs.items():
            for wav_chunk in wav_chunks:
                for class_name, labels in label_to_class_map.items():
                    if wav_chunk.label in labels:
                        wav_chunk.set_class(class_name)
                        break
                if wav_chunk.class_ is None:
                    raise ValueError("The label %s did not find a map to a class" % wav_chunk.label)
        classes = [wav_chunk.class_ for wav_chunks in self.wavs.values() for wav_chunk in wav_chunks]
        unique_classes = set(classes)
        logging.info("Mapped labels into the classes %s", unique_classes)
        for class_ in unique_classes:
            logging.info("Found that class %s has %d wav chunks", class_, classes.count(class_))

    def define_test_dataset(self, test_duration_sec_per_class=500):
        logging.info("Defining test datasets for each class")
        for class_ in self._get_classes():
            duration_sec_marked = 0
            wav_chunks = self._get_wavs_of_class(class_)
            wav_chunks.sort(key=lambda x: x.wav_id)
            for wav_chunk in wav_chunks:
                if duration_sec_marked + wav_chunk.duration_sec > test_duration_sec_per_class:
                    break
                wav_chunk.set_split('test')
                duration_sec_marked += wav_chunk.duration_sec
        
        wavs_of_test = self._get_wavs_of_split('test')
        test_duration_sec = sum([wav_chunk.duration_sec for wav_chunk in wavs_of_test])
        logging.info("%d wav chunks / %d duration seconds was defined as test dataset", len(wavs_of_test), test_duration_sec)
        for class_ in self._get_classes():
            wavs_of_test_of_class = [wav_chunk for wav_chunk in wavs_of_test if wav_chunk.class_ == class_]
            logging.info("Found %d wav chunks in test dataset of class %s", len(wavs_of_test_of_class), class_)


    def _get_classes(self):
        if not self.wavs:
            raise ValueError("No wav chunks loaded")
        classes = list(set([wav_chunk.class_ for wav_chunks in self.wavs.values() for wav_chunk in wav_chunks]))
        if None in classes:
            raise ValueError("Some wav chunks are not mapped to a class")
        return classes

    def _get_wavs_of_class(self, class_):
        return [wav_chunk for wav_chunks in self.wavs.values() for wav_chunk in wav_chunks if wav_chunk.class_ == class_]

    
    def _get_wavs_of_split(self, split):
        return [wav_chunk for wav_chunks in self.wavs.values() for wav_chunk in wav_chunks if wav_chunk.split == split]
    

    def define_training_dataset(self):
        logging.info("Defining training dataset")
        for _, wav_chunks in self.wavs.items():
            for wav_chunk in wav_chunks:
                if wav_chunk.split != "test":
                    wav_chunk.set_split('train')
        wavs_of_train = self._get_wavs_of_split('train')
        train_duration_sec = sum([wav_chunk.duration_sec for wav_chunk in wavs_of_train])
        logging.info("%d wav chunks / %d duration seconds was defined as train dataset", len(wavs_of_train), train_duration_sec)
        for class_ in self._get_classes():
            wavs_of_train_of_class = [wav_chunk for wav_chunk in wavs_of_train if wav_chunk.class_ == class_]
            logging.info("Found %d wav chunks in train dataset of class %s", len(wavs_of_train_of_class), class_)

    def add_augmentation_to_training_wavs(self, augmentations):
        if not augmentations:
            raise ValueError("No augmentations given")
        if self._get_wavs_with_augmentation():
            raise ValueError("Augmentations already added")
        
        for _, wav_chunks in self.wavs.items():
            augmentation_wavs = []
            for wav_chunk in wav_chunks:
                    if wav_chunk.split == 'train':
                        for augmentation in augmentations:
                            new_wav_chunk = wav_chunk.copy()
                            new_wav_chunk.add_augmentation(augmentation)
                            augmentation_wavs.append(new_wav_chunk)
            wav_chunks.extend(augmentation_wavs)


        wavs_with_augmentation = self._get_wavs_with_augmentation()
        logging.info("Added %d wav chunks with augmentations", len(wavs_with_augmentation))
        training_wavs = self._get_wavs_of_split('train')
        logging.info("Amount of wavs of training after adding augmentations: %d", len(training_wavs))

    
    def _get_wavs_with_augmentation(self):
        # Get all wavs with augmentation
        augmentation_wavs = []
        for _, wav_chunks in self.wavs.items():
            for wav_chunk in wav_chunks:
                if wav_chunk.split == 'train' and wav_chunk.augmentation is not None:
                    augmentation_wavs.append(wav_chunk)
        return augmentation_wavs
    
    def print_data_stats(self):
        logging.info("Logging data stats")
        for split in ['train', 'test']:
            for class_ in self._get_classes():
                wavs_of_class = [wav_chunk for wav_chunk in self._get_wavs_of_split(split) if wav_chunk.class_ == class_]
                logging.info("Found %d wav chunks in %s dataset of class %s", len(wavs_of_class), split, class_)

    
    def write_dataset(self, audio_format="logmel", clean=False, file_type="npy"):
        logging.info("Writing dataset")

        if clean and os.path.exists(self.output_path_to_data):
            shutil.rmtree(self.output_path_to_data)
        
        if not os.path.exists(self.output_path_to_data):
            os.mkdir(self.output_path_to_data)
        
        classes = self._get_classes()
        for split in ['train', 'test']:
            for class_ in classes:
                if not os.path.exists(os.path.join(self.output_path_to_data, split, class_)):
                    os.makedirs(os.path.join(self.output_path_to_data, split, class_))
        
        for _, wav_chunks in self.wavs.items():
            for wav_chunk in wav_chunks:
                wav_chunk.write(self.input_path_to_data, self.output_path_to_data, audio_format, file_type)
        






    
        
        
    


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



    
    
if __name__ == "__main__":
    # get_data()
    data = Data(os.getenv("DATA"))
    data.load_data()
    data.map_labels_to_classes({
        'drone': ['normal_drone', 'racing_drone', 'normal_fixedwing', 'petrol_fixedwing'],
        'non-drone': ['no_class']
    })

    data.define_test_dataset()
    data.define_training_dataset()
    data.add_augmentation_to_training_wavs(['low_pass', 'pitch_shift', 'add_noise', 'high_pass', 'band_pass'])
    data.print_data_stats()
    data.write_dataset(audio_format="stft", clean=True, file_type="npy")    












    








