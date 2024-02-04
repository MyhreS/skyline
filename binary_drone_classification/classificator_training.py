import sys
import platform
current_os = platform.system()
if current_os == 'Darwin':  # macOS
    sys.path.append('/Users/simonmyhre/workdir/gitdir/skyline')
elif current_os == 'Linux':  # Linux
    sys.path.append('/cluster/datastore/simonmy/skyline')
from dotenv import load_dotenv
import os
load_dotenv()

from cirrus import Data

from dotenv import load_dotenv
import os
import logging
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')

# Chech if GPU is available
logging.info("Num GPUs Available: %d", len(tf.config.experimental.list_physical_devices('GPU')))

data = Data(os.getenv("DATA_INPUT_PATH"), os.getenv("DATA_OUTPUT_PATH"))
data.window_it(1)
data.split_it(train_percent=70, test_percent=20, validation_percent=10)
data.label_to_class_map_it({
    'drone': ['normal_drone', 'racing_drone', 'normal_fixedwing', 'petrol_fixedwing'],
    'non-drone': ['no_class']
})
data.sample_rate_it(44100)
data.augment_it(['low_pass'])
data.audio_format_it('stft')
data.file_type_it('tfrecord')
data.limit_it(300)
#data.describe_it()
data.make_it()
train_ds, val_ds, test_ds, class_int_map, class_weights = data.load_it()



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
    layers.Input(shape=(1025, 87, 1)), # # 1 sec window: 128, 87. 2 sec window: 1025, 173.
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
    train_ds,
    validation_data=val_ds,
    epochs=1,
    verbose=2,
    callbacks = callbacks,
    class_weight=class_weights
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
test_loss, test_acc = model.evaluate(test_ds, verbose=2)
logging.info('Test Accuracy: %f', test_acc)





