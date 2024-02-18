import sys
import platform

current_os = platform.system()
if current_os == "Darwin":  # macOS
    sys.path.append("/Users/simonmyhre/workdir/gitdir/skyline")
elif current_os == "Linux":  # Linux
    sys.path.append("/cluster/datastore/simonmy/skyline")
import os
from dotenv import load_dotenv
from keras_tuner import RandomSearch, Hyperband
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

load_dotenv()
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M",
)
from cirrus import Data
from cumulus import Logger
from binary_drone_classification.search_model.cnn_model import CNNHyperModel

# Check if GPU is available
logging.info(
    "Num GPUs Available: %d", len(tf.config.experimental.list_physical_devices("GPU"))
)

"""
Data preprocessing
"""
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
# data.set_augmentations(['low_pass', 'high_pass', 'band_pass'])
data.set_audio_format("stft")
data.set_file_type("tfrecord")
data.set_limit(100)
# data.describe_it()
data.make_it()


"""
Train data
"""
train_ds, shape, class_weights = data.load_it(split="train", label_encoding="integer")
val_ds, shape = data.load_it(split="val", label_encoding="integer")

print(class_weights)
print(shape)

logger = Logger("run_1", clean=True)
length_train = sum(1 for _ in train_ds)
length_val = sum(1 for _ in val_ds)
data_config = {
    "Train length": length_train,
    "Val length": length_val,
    "Tensor shape": str(shape),
}
logger.save_data_config(data_config)


"""
Classificator
"""
cnn_hypermodel = CNNHyperModel(input_shape=(shape[0], shape[1], 1))
tuner = RandomSearch(
    cnn_hypermodel,
    objective="val_accuracy",
    max_trials=50,
    directory=logger.get_tuner_path(),
    project_name=logger.get_run_name(),
    max_model_size=8_000_000,
    overwrite=True,
    max_consecutive_failed_trials=50,
)
tuner.search(train_ds, validation_data=val_ds, epochs=2)
tuner.results_summary()
# best_model = tuner.get_best_models(num_models=1)[0]
