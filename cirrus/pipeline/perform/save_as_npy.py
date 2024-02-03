import os
import numpy as np

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')

def save_as_npy(spectrogram, output_path, file_name):
    # Save as npy
    path = os.path.join(output_path, file_name+".npy")
    logging.info("Writing %s", path)
    np.save(path, spectrogram)