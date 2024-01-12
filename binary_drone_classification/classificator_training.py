
from utils.utils import npy_spectrogram_dataset_from_directory

import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np



# Example Usage
train_dataset, class_names = npy_spectrogram_dataset_from_directory(
    'cache/train',
    #batch_size=32,
    label_mode='binary',
    validation_split=0.2,
    subset='training',
    seed=42,
    batch_size=32
)

val_dataset, class_names = npy_spectrogram_dataset_from_directory(
    'cache/train',
    #batch_size=32,
    label_mode='binary',
    validation_split=0.2,
    subset='validation',
    seed=42,
    batch_size=32
)

test_dataset, class_names = npy_spectrogram_dataset_from_directory(
    'cache/test',
    #batch_size=32,
    label_mode='binary',
    validation_split=None,
    subset=None,
    seed=42,
    batch_size=32
)

plt.figure(figsize=(10, 10))
for spectrograms, labels in train_dataset.take(1):
    for i in range(9):
        # Squeeze out the single-channel dimension
        spectrogram = np.squeeze(spectrograms[i].numpy())
        label = labels[i].numpy()

        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(spectrogram)
        plt.title(class_names[int(label)])
        plt.axis('off')

# Create a CNN model
model = tf.keras.Sequential([
    layers.Input(shape=(128, 87, 1)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Test
test_loss, test_acc = model.evaluate(test_dataset)
print('Test Accuracy:', test_acc)




