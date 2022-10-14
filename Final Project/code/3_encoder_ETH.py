import datetime

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from controllers.base_model import BaseModelFromScratch
from controllers.utils import set_seeds
from controllers.data import Data

set_seeds(0)


class Model(BaseModelFromScratch):
    def create_model(self):
        encoder = tf.keras.models.Sequential(name="encoder")

        encoder.add(
            tf.keras.layers.LSTM(
                126,
                return_sequences=True,
                input_shape=(self.dataset.X_train.shape[1:]),
            )
        )
        encoder.add(tf.keras.layers.LSTM(10, return_sequences=True))

        decoder = tf.keras.models.Sequential(name="decoder")
        decoder.add(tf.keras.layers.LSTM(64, return_sequences=True))
        decoder.add(
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.dataset.X_train.shape[-1], activation="sigmoid"))
        )

        self.model = tf.keras.models.Sequential([encoder, decoder])

    def train_model(self):
        self.model.compile(loss="mse", optimizer="Nadam")
        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard = tf.keras.callbacks.TensorBoard(f"{self.directory}/{self.project_name}/tb_logs/{time}")
        self.model.fit(
            self.dataset.X_train,
            self.dataset.X_train,
            validation_data=(self.dataset.X_validation, self.dataset.X_validation),
            epochs=750,
            batch_size=1000,
            callbacks=[tensorboard],
            shuffle=False,
        )


dataset = Data(
    lookback=10,
    train_size=0.6,
    validation_size=0.2,
    test_size=0.2,
    scaler=MinMaxScaler(),
    feature_set=f"ETH_Data",
    time_to_predict=1,
    features=None,
)

m = Model(
    model_name="ETH_autoencoder_model",
    dataset=dataset,
    directory="models_predict_1hr",
    project_name="ETH_autoencoder_model",
)
m.train()
m.save_model()
