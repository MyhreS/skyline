from keras.utils import plot_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    LeakyReLU,
    Flatten,
    Dense,
    MaxPooling2D,
    Dropout,
    Input,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50


shape = (63, 512)
# Load the base ResNet50 model
base_model = ResNet50(
    weights="imagenet", include_top=False, input_shape=(shape[0], shape[1], 3)
)
base_model.trainable = True

# Extract the output from the 80th layer and rename the model
layer_name = base_model.layers[80].name
resnet80 = Model(
    inputs=base_model.input,
    outputs=base_model.get_layer(layer_name).output,
    name="ResNet80",
)

# Create a new model, adding layers before and after the ResNet80
model = Sequential(
    [
        Input(shape=(shape[0], shape[1], 1)),
        Conv2D(3, (3, 3), padding="same"),
        resnet80,
        Conv2D(256, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(1, 2)),
        Dropout(0.5),
        Conv2D(256, (3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(1, 2)),
        Conv2D(256, (3, 3), padding="same", activation="relu"),
        Dropout(0.5),
        MaxPooling2D(pool_size=(1, 2)),
        Conv2D(256, (3, 3), padding="same", activation="relu"),
        Dropout(0.5),
        MaxPooling2D(pool_size=(1, 2)),
        Conv2D(256, (3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(1, 2)),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(128, activation="relu"),
        Dense(3, activation="softmax"),
    ]
)

# Plot the model architecture
plot_model(
    model, to_file="model_architecture.png", show_shapes=True, show_layer_names=True
)
