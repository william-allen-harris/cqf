"""
Module defining the Abstract Class Tensorflow model constructors.
"""

import os
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Type, Optional

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
import shap
import pandas as pd

from controllers.base_data import BaseData, DatasetEnum
from controllers.utils import save


@dataclass
class BaseModel(ABC):
    "Abstract class representing a tensorflow model constructor."
    dataset: Type[BaseData]
    model_name: str
    directory: str
    project_name: str

    def __post_init__(self):
        "Validate that the model path is available."
        self.check_folder(self.dir())

    def run_statistics(self, ds_enum: DatasetEnum):
        """
        Return validation statistics on a model
        """
        X, y = self.dataset.get(ds_enum)
        stats_pred = self.model.predict(X)
        y_argmax = [0 if x[0] < 0.5 else 1 for x in stats_pred]

        loss = log_loss(y, stats_pred, labels=[0, 1])
        f_1 = f1_score(y, y_argmax)
        recall = recall_score(y, y_argmax)
        precision = precision_score(y, y_argmax)
        accuracy = accuracy_score(y, y_argmax)
        auc = roc_auc_score(y, y_argmax)

        cm = confusion_matrix(y, y_argmax, labels=[0, 1], normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Sell", "Buy"])
        disp.plot()
        plt.savefig(
            self.dir(f"confusion_matrix_{ds_enum.name}.png"),
            dpi=100,
            bbox_inches="tight",
        )

        return {
            "log_val": loss,
            "accuracy_val": accuracy,
            "f_1": f_1,
            "recall": recall,
            "precision": precision,
            "auc": auc,
        }

    def shap_deepexplainer(self, sample_num: int = 100):
        "Display a shap deep explainer plot of a NN on aggregate."
        X100 = shap.utils.sample(self.dataset.X_train, sample_num, random_state=0)
        XTest100 = shap.utils.sample(self.dataset.X_test, sample_num, random_state=0)

        explainer = shap.DeepExplainer(self.model, X100)
        shap_values = explainer.shap_values(XTest100)

        feature_len = len(self.dataset.feature_names)
        shap_values_2D = shap_values[0].reshape(-1, feature_len)
        X_test_2D = XTest100.reshape(-1, feature_len)
        x_test_2d = pd.DataFrame(data=X_test_2D, columns=self.dataset.feature_names)

        plt.clf()

        shap.summary_plot(shap_values_2D, x_test_2d, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance Bar Plot {self.model_name}")
        plt.savefig(
            self.dir(f"SHAP_feature_bar_plot.png"),
            dpi=100,
            bbox_inches="tight",
        )
        plt.show()
        plt.close()
        shap.summary_plot(shap_values_2D, x_test_2d, show=False)
        plt.title(f"SHAP Feature Importance {self.model_name}")
        plt.savefig(
            self.dir(f"SHAP_feature_summary_plot.png"),
            dpi=100,
            bbox_inches="tight",
        )
        plt.show()
        plt.close()

    def shap_deepexplainer_per_looback(self, sample_num: int = 100):
        "Display a shap deep explainer plot of a NN at specific lookbacks."
        X100 = shap.utils.sample(self.dataset.X_train, sample_num, random_state=0)
        XTest100 = shap.utils.sample(self.dataset.X_test, sample_num, random_state=0)

        explainer = shap.DeepExplainer(self.model, X100)
        shap_values = explainer.shap_values(XTest100)

        feature_len = len(self.dataset.feature_names)
        shap_values_2D = shap_values[0].reshape(-1, feature_len)
        X_test_2D = XTest100.reshape(-1, feature_len)
        x_test_2d = pd.DataFrame(data=X_test_2D, columns=self.dataset.feature_names)

        len_test_set = len(X_test_2D)

        for step in range(self.dataset.lookback):
            indice = [i for i in list(range(len_test_set)) if i % self.dataset.lookback == step]
            shap_values_2D_step = shap_values_2D[indice]
            x_test_2d_step = x_test_2d.iloc[indice]
            print("---------------- Lookback {} ----------------".format(step))
            shap.summary_plot(shap_values_2D_step, x_test_2d_step, plot_type="bar")
            shap.summary_plot(shap_values_2D_step, x_test_2d_step)
            print("\n")

    def check_folder(self, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def dir(self, file_name: Optional[str] = ""):
        return os.path.join(self.directory, self.project_name, self.model_name, file_name)

    def save_model(self):
        "Save model to a cloudpickle file."
        save(self, self.dir("model.pickle"))


@dataclass
class BaseModelFromScratch(BaseModel, ABC):
    "Abstract Class representing a tensorflow model constructor."
    dataset: Type[BaseData]

    def train(self):
        self.create_model()
        self.train_model()

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def train_model(self):
        pass


@dataclass
class BaseModelFromHyperTuner(BaseModel):
    model: object
