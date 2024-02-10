import os
import numpy as np


def save_as_npy(spectrogram, output_path, file_name):
    # Save as npy
    path = os.path.join(output_path, file_name+".npy")
    np.save(path, spectrogram)