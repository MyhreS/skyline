import pandas as pd
import os
import shutil
import librosa
import numpy as np
from ...augmenter.augmenter import Augmenter
from .save_as_npy import save_as_npy
from .save_as_tfrecord import save_as_tfrecord

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')



def to_spectrogram(wav, sample_rate, spectrogram_type):
    spectrogram = None
    if spectrogram_type == 'logmel':
        mel_spectrogram = librosa.feature.melspectrogram(y=wav, sr=sample_rate, n_mels=128, n_fft=2048, hop_length=512, fmin=0, fmax=sample_rate // 2)
        spectrogram = librosa.power_to_db(mel_spectrogram)
    elif spectrogram_type == 'stft':
        stft = librosa.stft(wav, n_fft=2048, hop_length=512)
        spectrogram = np.abs(stft)
    else:
        raise ValueError("Invalid spectrogram_type. Choose 'logmel' or 'stft'.")
    return spectrogram
    

def pre_pipeline(df: pd.DataFrame, data_output_path: str, clean: bool):
    # Check if data_output_path exists (Here the wavs are going to be saved)
    if clean and os.path.exists(data_output_path):
        shutil.rmtree(data_output_path)
    
    if not os.path.exists(data_output_path):
        os.makedirs(data_output_path)

    data_output_path_contents = os.listdir(data_output_path)
    # Strip the file extension
    data_output_path_contents_hash = [file.split('.')[0] for file in data_output_path_contents]

    # Remove all files in df which are already in data_output_path
    len_before = len(df)
    wavs_to_pipeline_df = df[~df['hash'].isin(data_output_path_contents_hash)]
    len_after = len(wavs_to_pipeline_df)
    if len_before != len_after:
        logging.info("Skipping %s files which are already pipelined and saved in the data_output_path", len_before - len_after)
    return wavs_to_pipeline_df

def pipeline(df: pd.DataFrame, input_path: str, output_path: str):
    df = df.sort_values('wav_blob')
    df = df.reset_index(drop=True)

    wav_currently_read = None
    wav = None
    sample_rate = None
    augmenter = Augmenter()
    shape_validation = None

    for _, row in df.iterrows():
        # Read new wav if necessary
        if wav_currently_read != row['wav_blob']:
            wav_currently_read = row['wav_blob']
            wav, sample_rate = librosa.load(os.path.join(input_path, wav_currently_read), sr=44100)
        # Make a chunk of the wav
        wav_chunk = wav[int(row['label_relative_start_sec'] * sample_rate):int(row['label_relative_end_sec'] * sample_rate)]
        if row.get('augmentation') in augmenter.augment_options:
            wav_chunk_augmented = augmenter.augment(wav_chunk, sample_rate, row.get('augmentation'))
            wav_spectogram = to_spectrogram(wav_chunk_augmented, sample_rate, row['audio_format'])
        else:
            wav_spectogram = to_spectrogram(wav_chunk, sample_rate, row['audio_format'])

        # Validating that all outputted files have the same shape
        if shape_validation is None:
            shape_validation = wav_spectogram.shape
            logging.info("Writing files with shape %s", shape_validation)
        else:
            assert shape_validation == wav_spectogram.shape, "All outputted files must have the same shape"

        # Save the chunk
        if row['file_type'] == 'npy':
            save_as_npy(wav_spectogram, output_path, row['hash'])
        if row['file_type'] == 'tfrecord':
            save_as_tfrecord(wav_spectogram, output_path, row['hash'])


def post_pipeline(df: pd.DataFrame, data_info_output_path: str):
    if not os.path.exists(data_info_output_path):
        os.makedirs(data_info_output_path)
    df = df[['hash', 'class', 'split']]
    df.to_csv(os.path.join(data_info_output_path, 'dataset.csv'), index=False)
    

def perform(df: pd.DataFrame, data_input_path: str, data_output_path: str, clean=False):
    assert len(df) > 0, "Dataframe is empty"
    assert 'wav_blob' in df.columns, "Dataframe does not contain 'wav_blob' column"

    wavs_to_pipeline_df = pre_pipeline(df, data_output_path, clean)
    pipeline(wavs_to_pipeline_df, data_input_path, data_output_path)
    post_pipeline(df, "cache")
    






    
    

    

    
    
    




