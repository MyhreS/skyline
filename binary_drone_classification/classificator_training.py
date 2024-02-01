import logging
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


from utils.utils import npy_spectrogram_dataset_from_directory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')


# Chech if GPU is available
logging.info("Num GPUs Available: %d", len(tf.config.experimental.list_physical_devices('GPU')))

train_dataset, class_names = npy_spectrogram_dataset_from_directory(
    'cache/data/train',
    label_mode='binary',
    subset='training',
    seed=42,
    batch_size=32
)

val_dataset, class_names = npy_spectrogram_dataset_from_directory(
    'cache/data/validation',
    label_mode='binary',
    subset='validation',
    seed=42,
    batch_size=32
)

test_dataset, class_names = npy_spectrogram_dataset_from_directory(
    'cache/data/test',
    label_mode='binary',
    validation_split=None,
    subset=None,
    seed=42,
    batch_size=32
)

train_labels = np.concatenate([y for _, y in train_dataset], axis=0)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weight_dict = dict(enumerate(class_weights))
logging.info("Class weights: %s", class_weight_dict)

# plt.figure(figsize=(10, 10))
# for spectrograms, labels in train_dataset.take(1):
#     for i in range(9):
#         # Squeeze out the single-channel dimension
#         spectrogram = np.squeeze(spectrograms[i].numpy())
#         label = labels[i].numpy()

#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(spectrogram)
#         plt.title(class_names[int(label)])
#         plt.axis('off')

# Create a CNN model
model = tf.keras.Sequential([
    layers.Input(shape=(128, 87, 1)), # # 1 sec window: 128, 87. 2 sec window: 1025, 173.
    layers.Conv2D(32, 7, padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.Dropout(0.5),
    layers.Conv2D(32, 4, padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.Dropout(0.5),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.Dense(64),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.Dense(1, activation='sigmoid')
])



model.summary()

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.000005),
    loss='binary_crossentropy',
    metrics=['accuracy']
    )

callbacks = []
callbacks.append(EarlyStopping(monitor='val_loss', patience=10))


# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    verbose=2,
    callbacks = callbacks,
    class_weight=class_weight_dict
)

logging.info("History:")
logging.info("Train accuracy: %s", history.history["accuracy"])
logging.info("Train loss: %s", history.history["loss"])

# # Plot the training and validation accuracy
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.legend()
# plt.show()

# # Plot the training and validation loss
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.legend()
# plt.show()

# Test
test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
logging.info('Test Accuracy: %f', test_acc)





