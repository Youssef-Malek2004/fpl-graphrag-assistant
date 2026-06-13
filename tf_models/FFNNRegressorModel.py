import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable
import tensorflow as tf

# # Youssef's Config - Always Push with this one
# from tf_keras import layers, callbacks, regularizers
# import tf_keras as keras

# Mohamed's Config
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers

def _set_seed(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _build_ffnn(input_dim: int,
                hidden_units: Iterable[int] = (256, 128, 64),
                dropout: float = 0.10,
                l2: float = 1e-4,
                lr: float = 1e-3) -> keras.Model:
    reg = regularizers.l2(l2) if l2 and l2 > 0 else None
    model = keras.Sequential(name="ffnn_regressor")
    model.add(layers.Input(shape=(input_dim,)))
    for h in hidden_units:
        model.add(layers.Dense(h, activation="relu", kernel_regularizer=reg))
        if dropout and dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation="linear"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse"),
                 keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model


@dataclass
class FFNNRegressor:
    # Hyperparams
    hidden_units: Tuple[int, ...] = (512, 256, 128, 64, 32, 16)
    dropout: float = 0.10
    l2: float = 1e-4
    lr: float = 1e-3
    epochs: int = 500
    batch_size: int = 1024
    patience: int = 30
    seed: int = 42
    verbose: int = 1

    # Fitted artifacts
    model: Optional[keras.Model] = None

    def fit(self,
            X_train: np.ndarray, y_train: np.ndarray,
            X_valid: np.ndarray, y_valid: np.ndarray):
        _set_seed(self.seed)

        # Ensure numeric arrays + handle NaNs/Infs
        X_train = np.nan_to_num(np.asarray(X_train, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        X_valid = np.nan_to_num(np.asarray(X_valid, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
        y_valid = np.asarray(y_valid, dtype=np.float32).reshape(-1, 1)

        # Build & train
        self.model = _build_ffnn(
            input_dim=X_train.shape[1],
            hidden_units=self.hidden_units,
            dropout=self.dropout,
            l2=self.l2,
            lr=self.lr,
        )

        cbs = [
            callbacks.EarlyStopping(
                monitor="val_rmse", mode="min",
                patience=self.patience, restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_rmse", mode="min",
                factor=0.5, patience=max(5, self.patience // 3),
                min_lr=1e-6, verbose=1 if self.verbose else 0
            )
        ]

        self.model.fit(
            X_train, y_train,
            validation_data=(X_valid, y_valid),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=cbs,
            shuffle=True
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        preds = self.model.predict(X, verbose=0).reshape(-1)
        return preds