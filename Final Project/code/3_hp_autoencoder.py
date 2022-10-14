

from sklearn.preprocessing import MinMaxScaler
from controllers.data import Data
from controllers.hyper_tuner import HyperTuner
from controllers.utils import load
from controllers.base_data import Dataset

if __name__ == "__main__":
    for f_set in ['BTC', 'ETH']:
        dataset = Data(
                lookback=10,
                train_size=0.6,
                validation_size=0.2,
                test_size=0.2,
                scaler=MinMaxScaler(),
                feature_set=f"{f_set}_Data",
                time_to_predict=1,
                features=None,
            )
        autoencoder_model_class = load(f"/home/williamharris/Dev/CQF_final_project/models_predict_1hr/{f_set}_autoencoder_model/{f_set}_autoencoder_model/model.pickle")
        encoder_model = autoencoder_model_class.model.get_layer("encoder")

        encoded_dataset = Dataset(X_train=encoder_model.predict(dataset.X_train), y_train=dataset.y_train, 
                X_validation=encoder_model.predict(dataset.X_validation), y_validation=dataset.y_validation, 
            X_test=encoder_model.predict(dataset.X_test), y_test=dataset.y_test, feature_names=dataset.feature_names)

        tuner = HyperTuner(dataset=encoded_dataset, directory="models_predict_1hr", project_name=f"Autoencoder_{f_set}")
        tuner.search()
        tuner.save_hyper_tuner()
