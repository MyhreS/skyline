import tensorflow as tf
import os


def log_model(model: tf.keras.Model, run_id: str):
    path_to_model = os.path.join("cache", run_id, "model")
    if not os.path.exists(os.path.dirname(path_to_model)):
        os.makedirs(os.path.dirname(path_to_model))
    model.save(path_to_model)
