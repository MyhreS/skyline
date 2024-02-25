import sys
import platform

current_os = platform.system()
if current_os == "Darwin":  # macOS
    sys.path.append("/Users/simonmyhre/workdir/gitdir/skyline")
elif current_os == "Linux":  # Linux
    sys.path.append("/cluster/datastore/simonmy/skyline")
import os
import tensorflow as tf
from cirrus import Data
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M",
)

# Check if GPU is available
logging.info(
    "Num GPUs Available: %d", len(tf.config.experimental.list_physical_devices("GPU"))
)

data = Data(os.getenv("DATA_INPUT_PATH"), os.getenv("DATA_OUTPUT_PATH"))
data.set_window_size(1)
data.set_split_configuration(train_percent=65, test_percent=20, val_percent=15)
data.set_label_class_map(
    {
        "drone": [
            "normal_drone",
            "normal_fixedwing",
            "petrol_fixedwing",
            "racing_drone",
        ],
        "non-drone": ["nature_chernobyl", "false_positives_drone"],
    }
)
data.set_sample_rate(44100)
data.set_audio_format("stft")
data.set_file_type("tfrecord")
data.describe_it()
