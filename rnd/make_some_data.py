# PATH_TO_SKYLINE = "/cluster/datastore/simonmy/skyline"  # "/workspace/skyline"
# PATH_TO_INPUT_DATA = "/cluster/datastore/simonmy/data/datav3"  # "/workspace/data/data"
# PATH_TO_OUTPUT_DATA = (
#     "/cluster/datastore/simonmy/skyline/cache/data"  # "/workspace/skyline/cache/data"
# )
PATH_TO_SKYLINE = "/Users/simonmyhre/workdir/gitdir/skyline"
PATH_TO_INPUT_DATA = "/Users/simonmyhre/workdir/gitdir/sqml/projects/sm_multiclass_masters_project/pull_data/cache/datav3"
PATH_TO_OUTPUT_DATA = "/Users/simonmyhre/workdir/gitdir/skyline/cache/data"
import sys

sys.path.append(PATH_TO_SKYLINE)

from cirrus import Data

data = Data(PATH_TO_INPUT_DATA, PATH_TO_OUTPUT_DATA)
data.set_window_size(1)
data.set_val_of_train_split(0.2)
data.set_label_class_map(
    {
        "drone": [
            "electric_quad_drone",
            "racing_drone",
            "electric_fixedwing_drone",
            "petrol_fixedwing_drone",
        ],
        "non-drone": [
            "dvc_non_drone",
            "animal",
            "speech",
            "TUT_dcase",
            "nature_chernobyl",
        ],
    }
)
data.set_augmentations(["mix_1", "mix_2"], only_drone=True)
data.set_limit(100)
data.set_audio_format("log_mel")
data.describe_it()
data.make_it()
