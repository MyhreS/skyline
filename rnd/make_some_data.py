# PATH_TO_SKYLINE = "/cluster/datastore/simonmy/skyline"  # "/workspace/skyline"
# PATH_TO_INPUT_DATA = "/cluster/datastore/simonmy/data/datav3"  # "/workspace/data/data"
# PATH_TO_OUTPUT_DATA = (
#     "/cluster/datastore/simonmy/skyline/cache/data"  # "/workspace/skyline/cache/data"
# )
# PATH_TO_SKYLINE = "/Users/simonmyhre/workdir/gitdir/skyline"
# PATH_TO_INPUT_DATA = "/Users/simonmyhre/workdir/gitdir/sqml/projects/sm_multiclass_masters_project/pull_data/cache/datav3"
PATH_TO_SKYLINE = "/cluster/datastore/simonmy/skyline"
PATH_TO_INPUT_DATA = "/cluster/datastore/simonmy/data/datav3"
import os
import sys

sys.path.append(PATH_TO_SKYLINE)

from cirrus import Data

RUN_ID = "Test_some_data"
output_data = os.path.join("cache", RUN_ID, "data")
data = Data(PATH_TO_INPUT_DATA, output_data, RUN_ID)
data.set_window_size(2, load_cached_windowing=False)
data.set_val_of_train_split(0.2)
data.set_label_class_map(
    {
        "Data": [
            "racing_drone",
            "electric_fixedwing_drone",
            "petrol_fixedwing_drone",
            "electric_quad_drone",
            "dvc_non_drone",
            "animal",
            "speech",
            "TUT_dcase",
            "nature_chernobyl",
        ]
    }
)
# data.set_augmentations(["mix_1", "mix_2"], only_drone=True)
# data.set_limit(50)
data.set_audio_format("log_mel")
data.save_format("image")
data.describe_it()
data.make_it(clean=True)
