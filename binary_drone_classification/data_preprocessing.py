

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



def get_data():
    logging.info("Getting data")
    path_to_data = os.getenv("DATA")
    metadata_df = pd.read_csv(os.path.join(path_to_data, "data.csv"))
    logging.info("Shape of metadata: %s", metadata_df.shape)
    logging.info("Duration sec of metadata: %d", sum(metadata_df["duration_sec"]))

    concatenate_df = concatenate_same_labels_with_same_wav(metadata_df)

    windowed_df = window_labels_in_wavs(concatenate_df)

    split_df = train_test_split_wavs(windowed_df)

    classes_df = define_classes(split_df) 

    wav_chunk_pipeline(classes_df, path_to_data)


def concatenate_same_labels_with_same_wav(df):
    logging.info("Concatenating wavs")
    def concatenate_contiguous_rows(group):
        group = group.sort_values(by='relative_start_sec').reset_index(drop=True)
        new_rows = []
        i = 0
        while i < len(group):
            current_row = group.loc[i].copy()
            while i + 1 < len(group) and group.loc[i]['relative_end_sec'] == group.loc[i + 1]['relative_start_sec']:
                next_row = group.loc[i + 1]
                current_row['duration_sec'] += next_row['duration_sec']
                current_row['relative_end_sec'] = next_row['relative_end_sec']
                i += 1
            new_rows.append(current_row)
            i += 1
        return pd.DataFrame(new_rows)

    grouped = df.groupby(['wav_blob', 'label'], sort=False)
    concatenated_df = pd.concat([concatenate_contiguous_rows(group) for _, group in grouped])
    concatenated_df = concatenated_df.reset_index(drop=True)

    logging.info("Shape of metadata after concatenation: %s", concatenated_df.shape)
    logging.info("Duration sec of metadata after concatenation: %d", sum(concatenated_df["duration_sec"]))
    return concatenated_df
    

def window_labels_in_wavs(df):
    logging.info("Windowing labels in wavs")
    window_size = 2.0 # seconds

    def window_dataframe(df, window_size):
        windowed_rows = []

        for _, row in df.iterrows():
            start_sec = row['relative_start_sec']
            end_sec = row['relative_end_sec']
            
            while start_sec < end_sec:
                window_end_sec = min(start_sec + window_size, end_sec)
                new_row = row.copy()
                new_row['duration_sec'] = window_end_sec - start_sec
                new_row['relative_start_sec'] = start_sec
                new_row['relative_end_sec'] = window_end_sec
                windowed_rows.append(new_row)
                start_sec = window_end_sec

        return pd.DataFrame(windowed_rows)

    windowed_df = window_dataframe(df, window_size)
    windowed_df = windowed_df[windowed_df['duration_sec'] >= window_size]
    windowed_df = windowed_df.reset_index(drop=True)
    logging.info("Shape of metadata after windowing: %s", windowed_df.shape)
    logging.info("Duration sec of metadata after windowing: %d", sum(windowed_df["duration_sec"]))
    return windowed_df

def train_test_split_wavs(df):
    logging.info("Splitting wavs into train and test")
    window_size = 2.0 # seconds
    split_df = df.copy()
    split_df['split'] = 'train'
    split_df.loc[split_df.groupby('label').head(500/window_size).index, 'split'] = 'test'
    logging.info("Shape of train metadata after splitting: %s", split_df[split_df['split'] == 'train'].shape)
    logging.info("Duration sec of train metadata after splitting: %d", sum(split_df[split_df['split'] == 'train']["duration_sec"]))
    logging.info("Shape of test metadata after splitting: %s", split_df[split_df['split'] == 'test'].shape)
    logging.info("Duration sec of test metadata after splitting: %d", sum(split_df[split_df['split'] == 'test']["duration_sec"]))
    return split_df

def define_classes(df):
    logging.info("Defining classes based on labels")
    labels_to_classes = {
        'drone': ['normal_drone', 'racing_drone', 'normal_fixedwing', 'petrol_fixedwing'],
        'non-drone': ['no_class']
    }
    classes_df = df.copy()
    classes_df['class'] = pd.NA

    def map_label_to_class(row):
        for class_name, labels in labels_to_classes.items():
            if row['label'] in labels:
                return class_name
        return pd.NA

    classes_df['class'] = classes_df.apply(map_label_to_class, axis=1)

    if classes_df['class'].isna().any():
        raise ValueError("Some labels were not mapped to a class")
    return classes_df


def wav_chunk_pipeline(df, path_to_data):
    logging.info("Performing wav chunk pipeline")
    create_directories(df)

    def to_spectrogram(wav, sample_rate, spectrogram_type):
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
        
    def save_as_npy(spectrogram, wav_blob, split, class_, start_sec, end_sec, spectrogram_type, augment):
        wav_id = wav_blob.split('/')[-1].split('.')[0]
        save_dir_path = os.path.join("cache/data", split, class_)
        npy_filename = os.path.join(save_dir_path, f"{wav_id}_{start_sec}_{end_sec}_{spectrogram_type}_{augment}.npy")
        logging.info("Writing %s", npy_filename)
        np.save(npy_filename, spectrogram)

    def apply_low_pass_filter(wav, sample_rate, cutoff_freq=2000):
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = scipy.signal.butter(4, normal_cutoff, btype='low', analog=False)
        filtered_wav = scipy.signal.lfilter(b, a, wav)
        return filtered_wav
        

    def pitch_shift(wav, sample_rate, n_steps=0):
        return librosa.effects.pitch_shift(y=wav, sr=sample_rate, n_steps=n_steps)

    def add_noise(wav, noise_level=0.005):
        noise = np.random.randn(len(wav))
        augmented_wav = wav + noise_level * noise
        return np.clip(augmented_wav, -1.0, 1.0)

    
    def apply_high_pass_filter(wav, sample_rate, cutoff_freq=2000):
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = scipy.signal.butter(4, normal_cutoff, btype='high', analog=False)
        filtered_wav = scipy.signal.lfilter(b, a, wav)
        return filtered_wav
    
    def apply_band_pass_filter(wav, sample_rate, low_cutoff=500, high_cutoff=3000):
        nyquist = 0.5 * sample_rate
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        b, a = scipy.signal.butter(4, [low, high], btype='band', analog=False)
        filtered_wav = scipy.signal.lfilter(b, a, wav)
        return filtered_wav

    
    def augment_and_save(wav_chunk, sample_rate, row, spectrogram_type, augment_type):
        if augment_type == 'low_pass':
            augmented_wav_chunk = apply_low_pass_filter(wav_chunk, sample_rate)
        elif augment_type == 'pitch_shift':
            n_steps = random.uniform(-2, 2)
            augmented_wav_chunk = pitch_shift(wav_chunk, sample_rate, n_steps)
        elif augment_type == 'add_noise':
            augmented_wav_chunk = add_noise(wav_chunk)
        elif augment_type == 'high_pass':
            augmented_wav_chunk = apply_high_pass_filter(wav_chunk, sample_rate)
        elif augment_type == 'band_pass':
            augmented_wav_chunk = apply_band_pass_filter(wav_chunk, sample_rate)
        else:
            augmented_wav_chunk = wav_chunk  # No augmentation

        augmented_spectrogram = to_spectrogram(augmented_wav_chunk, sample_rate, spectrogram_type)
        save_as_npy(augmented_spectrogram, row['wav_blob'], row['split'], row['class'], row['relative_start_sec'], row['relative_end_sec'], spectrogram_type, augment_type)



    def pipeline(row):
        spectrogram_type = 'stft'
        split = row['split']

        wav, sample_rate = librosa.load(os.path.join(path_to_data, row['wav_blob']), sr=44100)
        wav_chunk = wav[int(row['relative_start_sec'] * sample_rate):int(row['relative_end_sec'] * sample_rate)]

        # Save original
        original_spectrogram = to_spectrogram(wav_chunk, sample_rate, spectrogram_type)
        save_as_npy(original_spectrogram, row['wav_blob'], split, row['class'], row['relative_start_sec'], row['relative_end_sec'], spectrogram_type, "none")

        # Apply augmentations
        if split == 'train':
            augmentations = ['low_pass', 'pitch_shift', 'add_noise', 'high_pass', 'band_pass']
            for augment in augmentations:
                augment_and_save(wav_chunk, sample_rate, row, spectrogram_type, augment)
    df.apply(pipeline, axis=1)
    print_data_stats(df)


def print_data_stats(df):
    logging.info("Logging data stats")

    # Get unique splits and classes
    splits = df['split'].unique()
    classes = df['class'].unique()

    # Get number of wavs per split and class
    for split in splits:
        for class_ in classes:
            number_of_wavs = os.listdir(os.path.join("cache/data", split, class_))
            logging.info("Number of wavs in %s %s: %d", split, class_, len(number_of_wavs))






def create_directories(df):
    logging.info("Creating directories")
    if not os.path.exists("cache"):
        os.mkdir("cache")

    if os.path.exists("cache/data"):
        shutil.rmtree("cache/data")
    os.mkdir("cache/data")

    splits = df['split'].unique()
    classes = df['class'].unique()
    for split in splits:
        os.mkdir(os.path.join("cache/data", split))
        for class_ in classes:
            os.mkdir(os.path.join("cache/data", split, class_))


if __name__ == "__main__":
    get_data()










    








