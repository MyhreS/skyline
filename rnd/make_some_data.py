PATH_TO_SKYLINE = "/cluster/datastore/simonmy/skyline"  # "/workspace/skyline"
PATH_TO_INPUT_DATA = "/cluster/datastore/simonmy/data/datav3"  # "/workspace/data/data"
PATH_TO_OUTPUT_DATA = (
    "/cluster/datastore/simonmy/skyline/cache/data"  # "/workspace/skyline/cache/data"
)
# PATH_TO_SKYLINE = "/Users/simonmyhre/workdir/gitdir/skyline"
# PATH_TO_INPUT_DATA = "/Users/simonmyhre/workdir/gitdir/sqml/projects/sm_multiclass_masters_project/pull_data/cache/data"
# PATH_TO_OUTPUT_DATA = "/Users/simonmyhre/workdir/gitdir/skyline/cache/data"
import sys

sys.path.append(PATH_TO_SKYLINE)

from cirrus import Data

data = Data(PATH_TO_INPUT_DATA, PATH_TO_OUTPUT_DATA)
# data.set_window_size(1)
# data.set_split_configuration(train_percent=50, test_percent=35, val_percent=15)
# data.set_label_class_map(
#     {
#         "drone": [
#             "normal_drone",
#             "normal_fixedwing",
#             "petrol_fixedwing",
#             "racing_drone",
#         ],
#         "non-drone": ["nature_chernobyl", "false_positives_drone"],
#     }
# )
# data.set_sample_rate(44100)
# data.set_augmentations(
#     ["low_pass", "pitch_shift", "add_noise", "high_pass", "band_pass"]
# )
# data.set_audio_format("log_mel")
# data.describe_it()
# data.make_it()
