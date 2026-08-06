from __future__ import annotations

from typing import Any

from tensorflow import keras


def build_ann(
    config: dict[str, Any],
    n_features: int,
    output_bias: float | None = None,
) -> keras.Model:
    layers = [
        keras.layers.Dense(
            config["hidden_layers"][0],
            activation="relu",
            input_shape=(n_features,),
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(config.get("dropout", 0.3)),
    ]
    for units in config["hidden_layers"][1:]:
        layers.append(keras.layers.Dense(units, activation="relu"))
        layers.append(keras.layers.BatchNormalization())
        layers.append(keras.layers.Dropout(config.get("dropout", 0.3)))

    bias_init = keras.initializers.Constant(output_bias) if output_bias else None
    layers.append(keras.layers.Dense(1, activation="sigmoid", bias_initializer=bias_init))

    model = keras.Sequential(layers)
    model.compile(
        optimizer=keras.optimizers.Adam(config.get("learning_rate", 0.001)),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="prc", curve="PR"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def build_autoencoder(
    config: dict[str, Any],
    n_features: int,
) -> keras.Model:
    encoding_dim = config.get("encoding_dim", 16)

    encoder = keras.Sequential(
        [
            keras.layers.Dense(64, activation="relu", input_shape=(n_features,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(encoding_dim, activation="relu"),
        ]
    )
    decoder = keras.Sequential(
        [
            keras.layers.Dense(32, activation="relu", input_shape=(encoding_dim,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(n_features, activation="linear"),
        ]
    )
    autoencoder = keras.Sequential([encoder, decoder])
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder
