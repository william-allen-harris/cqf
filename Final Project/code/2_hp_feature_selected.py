from controllers.data import Data
from controllers.features import get_scaler
from controllers.hyper_tuner import HyperTuner


if __name__ == "__main__":
    feature_sets = ["BTC", "ETH"]
    features = {
        "BTC": {
            "scaler": get_scaler(
                standscaler_cols=["BB_UPPER_10", "BB_LOWER_10", "BB_MIDDLE_10", "ATR_3", "HASH_RATE"],
                minmaxscaler_cols=["WILLIAMS_7", "STOCH_7", "WILLIAMS_3", "STOCH_3", "STOCH_21", "RSI_3", "CCI_5"],
            ),
            "features": [
                "BB_10",
                "ATR_3",
                "HASH_RATE",
                "WILLIAMS_7",
                "STOCH_7",
                "WILLIAMS_3",
                "STOCH_3",
                "STOCH_21",
                "RSI_3",
                "CCI_5",
            ],
        },
        "ETH": {
            "scaler": get_scaler(
                standscaler_cols=[
                    "DIFFICULTY",
                    "BLOCK_SIZE",
                ],
                minmaxscaler_cols=["STOCH_3", "STOCH_7", "PERCENT_B_5", "WILLIAMS_3", "WILLIAMS_32"],
            ),
            "features": [
                "DIFFICULTY",
                "BLOCK_SIZE",
                "STOCH_3",
                "STOCH_7",
                "PERCENT_B_5",
                "WILLIAMS_3",
                "WILLIAMS_32",
            ],
        },
    }

    for feature_set in feature_sets:
        dataset = Data(
            lookback=10,
            train_size=0.6,
            validation_size=0.2,
            test_size=0.2,
            feature_set=f"{feature_set}_Data",
            time_to_predict=1,
            **features[feature_set],
        )
        tuner = HyperTuner(dataset=dataset, directory="models_predict_1hr", project_name=f"Feature_selected_{feature_set}")
        tuner.search()
        tuner.save_hyper_tuner()
