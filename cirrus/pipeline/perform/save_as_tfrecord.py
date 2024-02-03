import os
import numpy as np
import tensorflow as tf
from typing import List
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')


def _floatList_feature(valuelist: List[float]):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=valuelist))

def _float_feature(value: float):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))




# def _bytes_feature(value):
#     """Returns a bytes_list from a string / byte."""
#     if isinstance(value, type(tf.constant(0))):
#         value = value.numpy() # BytesList won't unpack a string from an EagerTensor
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def save_as_tfrecord(spectrogram, output_path, file_name):
    # The path for the TFRecord file
    path = os.path.join(output_path, file_name + ".tfrecord")
    logging.info("Writing %s", path)
    # Create a TFRecordWriter
    with tf.io.TFRecordWriter(path) as writer:
        spectrogram_list = spectrogram.flatten().tolist()
        spectrogram_shape = spectrogram.shape
        # Create a feature
        feature = {
            'spectrogram': _floatList_feature(spectrogram_list),
            'shape': _int64_feature(list(spectrogram_shape)),
        }
        # Create an example
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write to file
        writer.write(example.SerializeToString())
