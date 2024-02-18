import tensorflow as tf
from tensorflow.keras import layers
from keras_tuner import HyperModel


class CNNHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = tf.keras.Sequential()

        model.add(layers.Input(shape=self.input_shape))

        # First Conv2D layer
        model.add(
            layers.Conv2D(
                filters=hp.Int("conv_1_filters", min_value=32, max_value=64, step=32),
                kernel_size=hp.Choice("conv_1_kernel", values=[3, 5]),
                padding="same",
            )
        )
        model.add(layers.LeakyReLU())
        model.add(
            layers.Dropout(
                rate=hp.Float("dropout_1", min_value=0.0, max_value=0.5, step=0.1)
            )
        )

        # Adding more Conv2D layers based on a range
        for i in range(hp.Int("num_conv_layers", 2, 5)):
            model.add(
                layers.Conv2D(
                    filters=hp.Int(
                        f"conv_{i + 2}_filters", min_value=64, max_value=256, step=64
                    ),
                    kernel_size=hp.Choice(f"conv_{i + 2}_kernel", values=[3, 5]),
                    padding="same",
                )
            )
            model.add(layers.LeakyReLU())
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(
                layers.Dropout(
                    rate=hp.Float(
                        f"dropout_{i + 2}", min_value=0.0, max_value=0.5, step=0.1
                    )
                )
            )

        # Flatten and Dense layers
        model.add(layers.Flatten())
        model.add(
            layers.Dense(
                units=hp.Int("dense_1_units", min_value=128, max_value=512, step=128)
            )
        )
        model.add(layers.LeakyReLU(alpha=0.01))
        model.add(layers.BatchNormalization())
        model.add(
            layers.Dropout(
                rate=hp.Float("dropout_dense_1", min_value=0.0, max_value=0.5, step=0.1)
            )
        )

        # Output layer
        model.add(layers.Dense(1, activation="sigmoid"))

        # Compile the model
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        return model
