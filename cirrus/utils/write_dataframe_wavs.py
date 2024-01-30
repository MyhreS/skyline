import pandas as pd
import os
import shutil
import librosa
import random
import scipy
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')

def create_temp_output_directory(output_path: str, splits: list, classes: list):
    split_path = output_path.split('/')
    split_path[-1] = 'temp_' + split_path[-1]
    temp_output_path = os.path.join(*split_path)

    if os.path.exists(temp_output_path):
        shutil.rmtree(temp_output_path)

    for split in splits:
        split_path = os.path.join(temp_output_path, split)
        for class_ in classes:
            class_path = os.path.join(split_path, class_)
            os.makedirs(class_path)
    return temp_output_path

def move_matching_old_files_to_temp_output_path(old_files: dict, df: pd.DataFrame, temp_output_path: str):
    for index, row in df.iterrows():
        if len(old_files) == 0:
            break
        if row['hash'] in old_files:
            # Move file to temp_output_path
            temp_output_path = os.path.join(temp_output_path, row['split'], row['class'])
            path_to_old_file = old_files[row['hash']]
            shutil.move(path_to_old_file, temp_output_path)
            # Remove the file from old_files
            del old_files[row['hash']]
            # Remove the file from the df
            df.drop(index, inplace=True)
    return df

def apply_low_pass_filter(wav, sample_rate, cutoff_freq=2000):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = scipy.signal.butter(4, normal_cutoff, btype='low', analog=False)
    filtered_wav = scipy.signal.lfilter(b, a, wav)
    return filtered_wav

def apply_pitch_shift(wav, sample_rate, n_steps=0):
    return librosa.effects.pitch_shift(y=wav, sr=sample_rate, n_steps=n_steps)

def apply_noise(wav, noise_level=0.005):
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
    

def augment(row: pd.Series, wav_chunk, sample_rate):
    if pd.isna(row['augmentation']):
        return wav_chunk
    if row['augmentation'] == 'low_pass':
        wav_chunk = apply_low_pass_filter(wav_chunk, sample_rate)
    elif row['augmentation'] == 'pitch_shift':
        n_steps = random.uniform(-2, 2)
        wav_chunk = apply_pitch_shift(wav_chunk, sample_rate, n_steps)
    elif row['augmentation'] == 'add_noise':
        wav_chunk = apply_noise(wav_chunk)
    elif row['augmentation'] == 'high_pass':
        wav_chunk = apply_high_pass_filter(wav_chunk, sample_rate)
    elif row['augmentation'] == 'band_pass':
        wav_chunk = apply_band_pass_filter(wav_chunk, sample_rate)
    else:
        raise ValueError(f"Invalid augmentation: {row['augmentation']}")
    return wav_chunk


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
    
def save(spectrogram, output_path):
    # Save as npy
    npy_filename = os.path.join(output_path+ ".npy")
    logging.info("Writing %s", npy_filename)
    np.save(npy_filename, spectrogram)
    
def pipeline(df: pd.DataFrame, input_path: str, output_path: str):
    df = df.sort_values('wav_blob')
    df = df.reset_index(drop=True)

    wav_currently_read = None
    wav = None
    sample_rate = None
    for _, row in df.iterrows():
        # Read new wav if necessary
        if wav_currently_read != row['wav_blob']:
            wav_currently_read = row['wav_blob']
            wav, sample_rate = librosa.load(os.path.join(input_path, wav_currently_read), sr=44100)
        # Make a chunk of the wav
        wav_chunk = wav[int(row['label_relative_start_sec'] * sample_rate):int(row['label_relative_end_sec'] * sample_rate)]
        wav_chunk_augmented = augment(row, wav_chunk, sample_rate)
        wav_spectogram = to_spectrogram(wav_chunk_augmented, sample_rate, row['audio_format'])
        # Save the chunk
        save(wav_spectogram, os.path.join(output_path, row['split'], row['class'], row['hash']))

def get_old_files(output_path: str):
    old_files = {}
    for root, dirs, files in os.walk(output_path):
        for file in files:
            old_files[file.split('.')[0]] = os.path.join(root, file)
    return old_files

def move_old_files(old_files: dict, temp_output_path: str):
    if not os.path.exists(temp_output_path):
        os.makedirs(temp_output_path)
    for hash_, path_to_old_file in old_files.items():
        shutil.move(path_to_old_file, os.path.join(temp_output_path, hash_ + '.npy'))

def create_directory_structure(unique_splits, unique_classes, output_path: str):
    for split in unique_splits:
        split_path = os.path.join(output_path, split)
        if not os.path.exists(split_path):
            os.makedirs(split_path)
        for class_ in unique_classes:
            class_path = os.path.join(split_path, class_)
            if not os.path.exists(class_path):
                os.makedirs(class_path)

def move_matching_old_files(df: pd.DataFrame, old_files: dict, output_path: str):
    rows_to_drop = []
    for index, row in df.iterrows():
        if row['hash'] in old_files:
            file_output_path = os.path.join(output_path, row['split'], row['class'], row['hash'] + '.npy')
            path_to_old_file = old_files[row['hash']]
            if os.path.exists(path_to_old_file):
                logging.info(f"Moving file from {path_to_old_file} to {file_output_path}")
                shutil.move(path_to_old_file, file_output_path)
                del old_files[row['hash']]
                rows_to_drop.append(index)
            else:
                logging.warning(f"File not found: {path_to_old_file}")
    df.drop(rows_to_drop, inplace=True)
    return df


def handle_old_files(output_path: str):
    old_files = get_old_files(output_path)
    split_path = output_path.split('/')
    split_path[-1] = 'temp_' + split_path[-1]
    temp_output_path = os.path.join(*split_path)
    move_old_files(old_files, temp_output_path)
    return temp_output_path, old_files

def pre_pipeline(df: pd.DataFrame, output_path: str): # TODO Move / reuse old files with similar hash
    # temp_output_path, old_files = handle_old_files(output_path)
    create_directory_structure(df['split'].unique(), df['class'].unique(), output_path)
    # df_excluding_matching_old_files = move_matching_old_files(df, old_files, output_path)
    # if os.path.exists(temp_output_path):
    #     shutil.rmtree(temp_output_path)
   # return df_excluding_matching_old_files

def write_dataframe_wavs(df: pd.DataFrame, input_path: str, output_path: str):
    assert len(df) > 0, "Dataframe is empty"
    assert 'wav_blob' in df.columns, "Dataframe does not contain 'wav_blob' column"

    pre_pipeline(df, output_path)
    pipeline(df, input_path, output_path)
    






    
    

    

    
    
    




