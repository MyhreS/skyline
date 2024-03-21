PATH_TO_SKYLINE = "/workspace/skyline"
PATH_TO_INPUT_DATA = "/workspace/data/datav3"
# PATH_TO_SKYLINE = "/cluster/datastore/simonmy/skyline"
# PATH_TO_INPUT_DATA = "/cluster/datastore/simonmy/data/datav3"

import os
import sys

sys.path.append(PATH_TO_SKYLINE)

from cirrus import Data


"""
Making the data for run_1
"""

RUN_ID_1 = "Run-1-drone-non_drone"
output_data = os.path.join("cache", RUN_ID_1, "data")
data_1 = Data(PATH_TO_INPUT_DATA, output_data, RUN_ID_1)
data_1.set_window_size(2, load_cached_windowing=True)
data_1.set_val_of_train_split(0.2)
data_1.set_label_class_map(
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
data_1.set_limit(500_000)
data_1.set_audio_format("log_mel")
data_1.save_format("image")
data_1.describe_it()
data_1.make_it(clean=True)


"""
Making the data for run_2
"""

RUN_ID_2 = "Run-2-electric_quad_drone-other_drone-non_drone"
output_data = os.path.join("cache", RUN_ID_2, "data")
data_2 = Data(PATH_TO_INPUT_DATA, output_data, RUN_ID_2)
data_2.set_window_size(2, load_cached_windowing=True)
data_2.set_val_of_train_split(0.2)
data_2.set_label_class_map(
    {
        "electric_quad_drone": ["electric_quad_drone"],
        "other-drones": [
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
data_2.set_limit(500_000)
data_2.set_audio_format("log_mel")
data_2.save_format("image")
data_2.describe_it()
data_2.make_it(clean=True)


"""
Making the data for run_3
"""

RUN_ID_3 = "Run-3-racing_drone-other_drone-non_drone"
output_data = os.path.join("cache", RUN_ID_3, "data")
data_3 = Data(PATH_TO_INPUT_DATA, output_data, RUN_ID_3)
data_3.set_window_size(2, load_cached_windowing=True)
data_3.set_val_of_train_split(0.2)
data_3.set_label_class_map(
    {
        "racing_drone": ["racing_drone"],
        "other-drones": [
            "electric_quad_drone",
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
data_3.set_limit(500_000)
data_3.set_audio_format("log_mel")
data_3.save_format("image")
data_3.describe_it()
data_3.make_it(clean=True)


"""
Making the data for run_4
"""

RUN_ID_4 = "Run-4-electric_fixedwing_drone-other_drone-non_drone"
output_data = os.path.join("cache", RUN_ID_4, "data")
data_4 = Data(PATH_TO_INPUT_DATA, output_data, RUN_ID_4)
data_4.set_window_size(2, load_cached_windowing=True)
data_4.set_val_of_train_split(0.2)
data_4.set_label_class_map(
    {
        "electric_fixedwing_drone": ["electric_fixedwing_drone"],
        "other-drones": [
            "electric_quad_drone",
            "racing_drone",
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
data_4.set_limit(500_000)
data_4.set_audio_format("log_mel")
data_4.save_format("image")
data_4.describe_it()
data_4.make_it(clean=True)


"""
Making the data for run_5
"""

RUN_ID_5 = "Run-5-petrol_fixedwing_drone-other_drone-non_drone"
output_data = os.path.join("cache", RUN_ID_5, "data")
data_5 = Data(PATH_TO_INPUT_DATA, output_data, RUN_ID_5)
data_5.set_window_size(2, load_cached_windowing=True)
data_5.set_val_of_train_split(0.2)
data_5.set_label_class_map(
    {
        "petrol_fixedwing_drone": ["petrol_fixedwing_drone"],
        "other-drones": [
            "electric_quad_drone",
            "racing_drone",
            "electric_fixedwing_drone",
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
data_5.set_limit(500_000)
data_5.set_audio_format("log_mel")
data_5.save_format("image")
data_5.describe_it()
data_5.make_it(clean=True)


"""
Making the data for run_6
"""

RUN_ID_6 = "Run-6-drone-other-speech"
output_data = os.path.join("cache", RUN_ID_6, "data")
data_6 = Data(PATH_TO_INPUT_DATA, output_data, RUN_ID_6)
data_6.set_window_size(2, load_cached_windowing=True)
data_6.set_val_of_train_split(0.2)
data_6.set_label_class_map(
    {
        "drone": [
            "electric_quad_drone",
            "racing_drone",
            "electric_fixedwing_drone",
            "petrol_fixedwing_drone",
        ],
        "other": ["dvc_non_drone", "animal", "TUT_dcase", "nature_chernobyl"],
        "speech": ["speech"],
    }
)
data_6.set_limit(500_000)
data_6.set_audio_format("log_mel")
data_6.save_format("image")
data_6.describe_it()
