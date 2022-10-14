from dataclasses import dataclass

import keras_tuner
import tensorflow as tf
import numpy as np
from sklearn.utils import class_weight

from controllers.base_hyper_tuner import BaseHyperTuner
from controllers.utils import set_seeds

set_seeds(0)


@dataclass
class HyperTuner(BaseHyperTuner):
    @property
    def tuner(self):
        return keras_tuner.BayesianOptimization(
            hypermodel=self.build_model,
            objective=keras_tuner.Objective("val_binary_accuracy", direction="max"),
            max_trials=50,
            overwrite=False,
            seed=0,
            directory=self.directory,
            project_name=self.project_name,
        )

    def build_model(self, hp):
        model = tf.keras.models.Sequential()
        dropout = hp.Float(f"dropout", min_value=0, max_value=0.4, step=0.1)
        model.add(
            tf.keras.layers.LSTM(
                return_sequences=True,
                units=hp.Int(f"units_1", min_value=32, max_value=320),
                dropout=dropout,
                input_shape=(self.dataset.X_train.shape[1:]),
            )
        )

        for i in range(hp.Int("num_layers", 1, 6)):
            model.add(
                tf.keras.layers.LSTM(
                    return_sequences=True,
                    units=hp.Int(f"units_{i+1}", min_value=32, max_value=320),
                    dropout=dropout,
                )
            )
        model.add(tf.keras.layers.LSTM(units=hp.Int(f"units_last", min_value=32, max_value=320), dropout=dropout))
        model.add(
            tf.keras.layers.Dense(
                hp.Int(f"dense_units", min_value=32, max_value=128),
                activation="relu",
            )
        )
        model.add(tf.keras.layers.Dropout(hp.Float("dropout_last", min_value=0.1, max_value=0.4, step=0.1)))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

        model.compile(
            optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(), tf.keras.metrics.BinaryAccuracy()],
        )
        return model

    def search(self):
        print(self.tuner.search_space_summary())
        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        tensorboard = tf.keras.callbacks.TensorBoard(f"{self.directory}/{self.project_name}/tb_logs")
        class_weights = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.unique(self.dataset.y_train.tolist()),
            y=[i for x in self.dataset.y_train for i in x],
        )
        class_weights = dict(enumerate(class_weights))
        print(f"Class Weights {class_weights}")
        self.tuner.search(
            self.dataset.X_train,
            self.dataset.y_train,
            batch_size=512,
            epochs=100,
            callbacks=[early_stop, tensorboard],
            class_weight=class_weights,
            validation_data=(self.dataset.X_validation, self.dataset.y_validation),
            shuffle=False,
        )
