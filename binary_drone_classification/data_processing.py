
import os
import shutil
from dotenv import load_dotenv
import logging
import pandas as pd
import librosa
import numpy as np

logging.basicConfig(level=logging.INFO)
load_dotenv()

import librosa
import numpy as np
import os

def data_pipeline(wav_dir_path, save_dir_path, metadata, spectrogram_type='logmel'):
    n_fft = 2048  # Window size for STFT or Mel spectrogram
    hop_length = 512  # Hop length for STFT or Mel spectrogram
    n_mels = 128  # Increase for higher frequency resolution
    fmin = 0  # Minimum frequency for Mel spectrogram
    fmax = 44100 // 2  # Maximum frequency for Mel spectrogram

    os.makedirs(save_dir_path, exist_ok=True)

    for wav_id in metadata["wav_id"]:
        wav_file_path = os.path.join(wav_dir_path, wav_id + ".wav")
        audio_data, sample_rate = librosa.load(wav_file_path, sr=44100)
        chunk_size = 44100

        num_chunks = len(audio_data) // chunk_size
        for i in range(num_chunks):
            chunk = audio_data[i * chunk_size:(i + 1) * chunk_size]

            if spectrogram_type == 'logmel':
                # Compute log-Mel spectrogram
                mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax)
                spectrogram = librosa.power_to_db(mel_spectrogram)
            elif spectrogram_type == 'stft':
                # Compute STFT spectrogram
                stft = librosa.stft(chunk, n_fft=n_fft, hop_length=hop_length)
                spectrogram = np.abs(stft)
            else:
                raise ValueError("Invalid spectrogram_type. Choose 'logmel' or 'stft'.")

            npy_filename = os.path.join(save_dir_path, f"{wav_id}_{i}.npy")
            np.save(npy_filename, spectrogram)



def training(amount=None):
    """
    Get the data used for training and validation
    """

    # Get the training data
    path_to_training_data = os.getenv("TRAIN")
    train_metadata = pd.read_csv(os.path.join(path_to_training_data, "training_data.csv"))
    print(train_metadata.shape)

    # Get the rows where the column "drone" is true, and speech, bird and car is false
    drone_true_metadata = train_metadata[(train_metadata["drone"] == True) & (train_metadata["speech"] == False) & (train_metadata["bird"] == False) & (train_metadata["car"] == False)]
    print(drone_true_metadata.shape)
    print(drone_true_metadata.head())

    # Get the rows where alle the columns are false
    drone_false_metadata = train_metadata[(train_metadata["drone"] == False) & (train_metadata["speech"] == False) & (train_metadata["bird"] == False) & (train_metadata["car"] == False)]
    print(drone_false_metadata.shape)
    print(drone_false_metadata.head())

    # Print the duration_sec of the two dataframes
    print(sum(drone_true_metadata["duration_sec"]))
    print(sum(drone_false_metadata["duration_sec"]))

    # Create cache folder for storage of files
    if os.path.exists("cache/train"):
        shutil.rmtree("cache/train")
    if not os.path.exists("cache"):
        os.mkdir("cache")
    os.mkdir("cache/train")

    # Run the data_pipeline function on the drone folder
    wav_dir_path = os.path.join(path_to_training_data, "wavs")

    if amount:
        drone_true_metadata = drone_true_metadata[:amount]
        drone_false_metadata = drone_false_metadata[:amount]
    data_pipeline(wav_dir_path, "cache/train/drone", drone_true_metadata, spectrogram_type='logmel')
    data_pipeline(wav_dir_path, "cache/train/non_drone", drone_false_metadata, spectrogram_type='logmel')


    """
    Balance the dataset by making them equal in size.
    """
    files_in_drone = os.listdir("cache/train/drone")
    files_in_non_drone = os.listdir("cache/train/non_drone")

    # Cut the list so they are the same length
    if len(files_in_drone) > len(files_in_non_drone):
        files_in_drone = files_in_drone[:len(files_in_non_drone)]
    else:
        files_in_non_drone = files_in_non_drone[:len(files_in_drone)]

    # Remove the files that are not in the cut list
    for file in os.listdir("cache/train/drone"):
        if file not in files_in_drone:
            os.remove(os.path.join("cache/train/drone", file))
    for file in os.listdir("cache/train/non_drone"):
        if file not in files_in_non_drone:
            os.remove(os.path.join("cache/train/non_drone", file))
    

def testing(amount=None):
    """
    Get the data used for testing
    """

    # Get the testing data
    path_to_testing_data = os.getenv("TEST")
    test_metadata = pd.read_csv(os.path.join(path_to_testing_data, "testing_data.csv"))
    print(test_metadata.shape)

    # Get the rows where the column "drone" is true, and speech, bird and car is false
    drone_true_metadata = test_metadata[(test_metadata["drone"] == True) & (test_metadata["speech"] == False) & (test_metadata["bird"] == False) & (test_metadata["car"] == False)]
    print(drone_true_metadata.shape)
    print(drone_true_metadata.head())

    # Get the rows where alle the columns are false
    drone_false_metadata = test_metadata[(test_metadata["drone"] == False) & (test_metadata["speech"] == False) & (test_metadata["bird"] == False) & (test_metadata["car"] == False)]
    print(drone_false_metadata.shape)
    print(drone_false_metadata.head())

    # Print the duration_sec of the two dataframes
    print(sum(drone_true_metadata["duration_sec"]))
    print(sum(drone_false_metadata["duration_sec"]))

    # Create cache folder for storage of files
    if os.path.exists("cache/test"):
        shutil.rmtree("cache/test")
    if not os.path.exists("cache"):
        os.mkdir("cache")
    os.mkdir("cache/test")

    # Run the data_pipeline function on the drone folder
    wav_dir_path = os.path.join(path_to_testing_data, "wavs")
    if amount:
        drone_true_metadata = drone_true_metadata[:amount]
        drone_false_metadata = drone_false_metadata[:amount]
    data_pipeline(wav_dir_path, "cache/test/drone", drone_true_metadata, spectrogram_type='logmel')
    data_pipeline(wav_dir_path, "cache/test/non_drone", drone_false_metadata, spectrogram_type='logmel')


    """
    Balance the dataset by making them equal in size.
    """
    files_in_drone = os.listdir("cache/test/drone")
    files_in_non_drone = os.listdir("cache/test/non_drone")

    # Cut the list so they are the same length
    if len(files_in_drone) > len(files_in_non_drone):
        files_in_drone = files_in_drone[:len(files_in_non_drone)]
    else:
        files_in_non_drone = files_in_non_drone[:len(files_in_drone)]
    
    # Remove the files that are not in the cut list
    for file in os.listdir("cache/test/drone"):
        if file not in files_in_drone:
            os.remove(os.path.join("cache/test/drone", file))
    for file in os.listdir("cache/test/non_drone"):
        if file not in files_in_non_drone:
            os.remove(os.path.join("cache/test/non_drone", file))

if __name__ == "__main__":
    training(10)
    testing(10)

    









    








