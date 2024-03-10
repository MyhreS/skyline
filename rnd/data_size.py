PATH_TO_SKYLINE = "/cluster/datastore/simonmy/skyline"  # "/workspace/skyline"
PATH_TO_INPUT_DATA = "/cluster/datastore/simonmy/data/datav1"  # "/workspace/data/data"
PATH_TO_OUTPUT_DATA = (
    "/cluster/datastore/simonmy/skyline/cache/data"  # "/workspace/skyline/cache/data"
)
import sys

sys.path.append(PATH_TO_SKYLINE)

import logging
from cirrus import Data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M",
)

data = Data(PATH_TO_INPUT_DATA, PATH_TO_OUTPUT_DATA)
data.set_window_size(1)
data.set_split_configuration(train_percent=50, test_percent=35, val_percent=15)
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
data.set_audio_format("log_mel")
data.set_file_type("tfrecord")
data.describe_it()
