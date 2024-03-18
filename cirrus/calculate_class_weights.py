import numpy as np


def calculate_class_weights(dataset):
    """
    Calculate class weights for a given dataset and print the number of samples per class.

    Args:
    dataset (tf.data.Dataset): The dataset for which to calculate class weights.

    Returns:
    dict: A dictionary containing the class weights where keys are class indices and values are the weights.
    """
    class_counts = {}

    for image, label in dataset.unbatch().take(-1):
        label = label.numpy()
        if type(label) != np.int32:
            if len(label) == 1:  # Binary
                label = int(label[0])
            else:
                label = label.argmax()  # One-hot

        # Count the label
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1

    # Print the class counts
    for class_index, count in class_counts.items():
        print(f"Class {class_index} contains {count} samples.")

    # Calculate the class weights
    total_samples = sum(class_counts.values())
    class_weights = {
        class_id: total_samples / (len(class_counts) * count)
        for class_id, count in class_counts.items()
    }
    print(f"Class weights:\n{class_weights}")
    return class_weights
