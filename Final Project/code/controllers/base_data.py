"""
Module defining the Data class.
"""
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from toolz.sandbox import unzip

from controllers.features import Features


class DatasetEnum(Enum):
    Train = 1
    Validation = 2
    Test = 3


@dataclass
class Dataset:
    X_train: np.array
    y_train: np.array
    X_validation: np.array
    y_validation: np.array
    X_test: np.array
    y_test: np.array
    feature_names: list

    def get(self, dataset: DatasetEnum):
        "return the dataset for a particular DataSet enum."

        if not isinstance(dataset, DatasetEnum):
            raise ValueError("dataset must be of an enum in Dataset.")

        if dataset.name == "Train":
            return self.X_train, self.y_train
        elif dataset.name == "Validation":
            return self.X_validation, self.y_validation
        elif dataset.name == "Test":
            return self.X_test, self.y_test

        raise Exception("Unknown enum DataType.")


@dataclass
class BaseData(ABC):
    """
    Abstract class defining a Dataclass.
    This class translates a dataframe holding features and a TARGET column,
    into training, testing and validation numpy array sets.
    """

    lookback: int
    scaler: object
    features: Optional[list]
    train_size: float
    validation_size: float
    test_size: float

    @abstractmethod
    def dataframe(self) -> pd.DataFrame:
        """
        Return a dataframe holding features and a TARGET column.
        This will be saved to the class variable self.df.
        """
        return

    def __post_init__(self):
        self.df = self.dataframe()
        if "TARGET" not in self.df.columns:
            raise ValueError("Please specify a TARGET column!")

        train_df, validation_df, test_df = self.__split(self.df, self.train_size, self.validation_size, self.test_size)

        train_df = Features.add_features(train_df, self.features)
        validation_df = Features.add_features(validation_df, self.features)
        test_df = Features.add_features(test_df, self.features)

        feature_names = list(train_df.drop("TARGET", axis=1).columns)

        if not (list(train_df.columns) == list(validation_df.columns) == list(test_df.columns)):
            raise Exception("Error in adding features all dataset columns are not the same!")

        self.__validate_column_transformer(train_df)
        self.__validate_column_transformer(validation_df)
        self.__validate_column_transformer(test_df)

        __x_train, __y_train, self.train_df = self.__gen_seq(train_df, is_train_set=True)
        __x_validation, __y_validation, self.val_df = self.__gen_seq(validation_df, is_train_set=False)
        __x_test, __y_test, self.test_df = self.__gen_seq(test_df, is_train_set=False)

        self.dataset = Dataset(
            X_train=__x_train,
            y_train=__y_train,
            X_validation=__x_validation,
            y_validation=__y_validation,
            X_test=__x_test,
            y_test=__y_test,
            feature_names=feature_names,
        )
        self.__get_stats(train_df, self.dataset.y_train, "X_train")
        self.__get_stats(validation_df, self.dataset.y_validation, "X_validation")
        self.__get_stats(test_df, self.dataset.y_test, "X_test")

    def __validate_column_transformer(self, df):
        if isinstance(self.scaler, ColumnTransformer):
            scaler_cols = [i for x in self.scaler.transformers for i in x[2]]
            if not all(col in df.columns for col in scaler_cols):
                raise ValueError("Feature in ColumnTransformer is not in features")
            if not all(col in scaler_cols for col in df.columns if col != "TARGET"):
                raise ValueError("Feature in features is not in ColumnTransformer")

    @staticmethod
    def __get_stats(df: pd.DataFrame, y_dataset: np.array, name: str):
        "Print a dataset statistics"
        print(f"--------------------- {name} ---------------------")
        print(f"Features: {list(df.columns)}")
        print(f"start date: {df.index.min()}, end date: {df.index.max()}")
        values, counts = np.unique(y_dataset, return_counts=True)
        counts_dict = dict(zip(values, counts))
        counts_dict = {key: value/counts.sum() for key, value in counts_dict.items()}
        print(f"TARGET Percentages: {counts_dict}")

    @staticmethod
    def __split(df, train_size: float, validation_size: float, test_size: float):
        "Split the dataframe data df into training, testing and validation Datasets."
        if round(train_size + validation_size + test_size, 1) != 1.0:
            raise ValueError("train_size, validation_size and test_size must sum to zero.")

        total_length = len(df)
        train_index = int(total_length * train_size)
        validation_index = int(total_length * (validation_size + train_size))

        train_df = df.iloc[:train_index]
        validation_df = df.iloc[train_index:validation_index]
        test_df = df.iloc[validation_index:]

        return train_df, validation_df, test_df

    def __gen_seq(self, df: pd.DataFrame, is_train_set: bool):
        "Generate the X and y sequence Datasets."

        input_x = df.drop("TARGET", axis=1)
        input_x = self.scaler.fit_transform(input_x) if is_train_set else self.scaler.transform(input_x)
        input_y = df["TARGET"].values

        seq = []
        prev_days = deque(maxlen=self.lookback)

        for x, y, dt in zip(input_x, input_y, df.index):
            prev_days.append(x)
            if len(prev_days) == self.lookback:
                seq.append([np.array(prev_days), y, dt])

        x, y, dt = unzip(seq)
        return (np.array(list(x)), np.array(list([i] for i in y)), pd.DataFrame(list(dt)))

    @property
    def feature_names(self):
        return self.dataset.feature_names

    @property
    def X_train(self):
        return self.dataset.X_train

    @property
    def y_train(self):
        return self.dataset.y_train

    @property
    def X_test(self):
        return self.dataset.X_test

    @property
    def y_test(self):
        return self.dataset.y_test

    @property
    def X_validation(self):
        return self.dataset.X_validation

    @property
    def y_validation(self):
        return self.dataset.y_validation

    def get(self, dataset_enum: DatasetEnum):
        "return the dataset for a particular DataSet enum."
        return self.dataset.get(dataset_enum)
