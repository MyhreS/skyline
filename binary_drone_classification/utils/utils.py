import numpy as np
import tensorflow as tf
import os
import logging


def npy_spectrogram_dataset_from_directory(directory, label_mode='binary', validation_split=None, subset=None, seed=None, batch_size=None):
    # Print
    if subset:
        logging.info("Loading subset: %s", subset)

    # Helper function to load data from .npy files
    def load_npy_files(subdir, label):
        data = []
        labels = []
        subdir_path = os.path.join(directory, subdir)
        for filename in os.listdir(subdir_path):
            if filename.endswith('.npy'):
                filepath = os.path.join(subdir_path, filename)
                spectrogram = np.load(filepath)
                # Add a channel dimension for grayscale
                spectrogram = np.expand_dims(spectrogram, axis=-1)
                data.append(spectrogram)
                labels.append(label)
        return np.array(data), np.array(labels)
    
    # Check if the directory structure is valid
    if not os.path.isdir(directory) or not os.listdir(directory):
        raise ValueError(f"Could not find directory {directory} or directory is empty.")

    # Identify subdirectories (classes)
    classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    classes.sort()

    # Load data and labels
    data = []
    labels = []
    for index, class_name in enumerate(classes):
        class_data, class_labels = load_npy_files(class_name, index)
        data.append(class_data)
        labels.append(class_labels)
    
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Shuffle the dataset
    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    # Handle binary or categorical labels
    if label_mode == 'categorical':
        labels = tf.keras.utils.to_categorical(labels, num_classes=len(classes))

    # Split data into training and validation sets
    if validation_split and subset:
        validation_samples = int(data.shape[0] * validation_split)
        if subset == 'training':
            data = data[:-validation_samples]
            labels = labels[:-validation_samples]
        elif subset == 'validation':
            data = data[-validation_samples:]
            labels = labels[-validation_samples:]

    # Create TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))

    # Batch the dataset
    if batch_size:
        dataset = dataset.batch(batch_size)
    
    logging.info("Found %d files belonging to %d classes.", len(data), len(classes))
    logging.info("Shape of data: %s", data.shape)
    logging.info("Shape of labels: %s", labels.shape)


    return dataset, classes
