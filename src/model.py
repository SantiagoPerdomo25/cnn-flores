"""Definición de la CNN (con Data Augmentation y normalización integradas)."""
from tensorflow import keras
from tensorflow.keras import layers


def build_augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")


def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return layers.MaxPooling2D()(x)


def build_cnn(input_shape=(180, 180, 3), num_classes=5, augment=True):
    inputs = keras.Input(shape=input_shape)
    x = build_augmentation()(inputs) if augment else inputs  # solo activo en entrenamiento
    x = layers.Rescaling(1.0 / 255)(x)                        # normalización a [0, 1]
    for filters in (32, 64, 128, 256):
        x = conv_block(x, filters)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="cnn_flores")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
