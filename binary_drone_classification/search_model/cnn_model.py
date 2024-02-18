import tensorflow as tf
from tensorflow.keras import Model
from keras_tuner import HyperModel

# import the different layers I use
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    LeakyReLU,
    Dropout,
    MaxPooling2D,
    Flatten,
    Dense,
    BatchNormalization,
)


class CNNHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):

        input_layer = Input(shape=self.input_shape)

        # Main CNN layer 1
        x = Conv2D(
            filters=hp.Int("conv_1_filters", min_value=32, max_value=256, step=32),
            kernel_size=(3, 3),
            padding="same",
        )(input_layer)
        x = LeakyReLU()(x)
        x = Dropout(
            rate=hp.Float("dropout_conv1", min_value=0.0, max_value=0.4, step=0.2)
        )(x)
        x = BatchNormalization()(x)

        # Dynamic number of convolutional layers
        for i in range(
            1, hp.Int("num_conv_blocks", 3, 8)
        ):  # Adjust the range as needed
            x = Conv2D(
                filters=hp.Int(
                    f"conv_{i + 1}_filters", min_value=64, max_value=256, step=32
                ),
                kernel_size=(3, 3),
                padding="same",
            )(x)
            x = LeakyReLU()(x)
            x = Dropout(
                rate=hp.Float(
                    f"dropout_conv{i + 1}", min_value=0.0, max_value=0.4, step=0.2
                )
            )(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

        # Final CNN layer before flattening
        x = Conv2D(filters=256, kernel_size=(3, 3), padding="same")(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)

        # Dense layers
        x = Dense(units=hp.Int("dense_1_units", min_value=128, max_value=512, step=64))(
            x
        )
        x = LeakyReLU(alpha=0.01)(x)
        x = Dropout(
            rate=hp.Float("dropout_dense1", min_value=0.0, max_value=0.5, step=0.2)
        )(x)

        # Output layer
        output = Dense(1, activation="sigmoid")(x)

        # Construct model
        model = Model(inputs=input_layer, outputs=output)

        # Compile the model
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        return model
