import pandas as pd
import os
import numpy as np
import tensorflow as tf
from typing import List


def _floatList_feature(valuelist: List[float]):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=valuelist))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_as_tfrecord(
    audio: np.ndarray,
    output_path: str,
    file_name: str,
    audio_format: str,
):
    path = os.path.join(output_path, file_name + ".tfrecord")

    audio_shape = audio.shape
    if audio_format != "waveform":
        audio = audio.flatten().tolist()
    else:
        audio = audio.tolist()

    with tf.io.TFRecordWriter(path) as writer:
        feature = {
            "audio": _floatList_feature(audio),
            "shape": _int64_feature(list(audio_shape)),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
