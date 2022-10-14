from itertools import product


from controllers.data import Data
from controllers.features import get_scaler
from controllers.hyper_tuner import HyperTuner


if __name__ == "__main__":
    lookbacks = [10, 30, 60]
    feature_set = ["BTC", "ETH"]
    product_list = list(product(lookbacks, feature_set))

    comb = {}
    for p in product_list:
        dataset = Data(
            lookback=p[0],
            train_size=0.6,
            validation_size=0.2,
            test_size=0.2,
            scaler=get_scaler(),
            feature_set=f"{p[1]}_Data",
            time_to_predict=1,
            features=None,
        )
        tuner = HyperTuner(dataset=dataset, directory="models_predict_1hr", project_name=f"Base_{p[1]}_LB_{p[0]}")
        tuner.search()
        tuner.save_hyper_tuner()
