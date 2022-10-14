"""
Module defining the Abstract Class BaseHyperTuner
"""
import os
from typing import Optional
from abc import ABC, abstractmethod, abstractproperty, abstractstaticmethod
from dataclasses import dataclass

from typing import Type

from controllers.base_data import BaseData
from controllers.utils import save


@dataclass
class BaseHyperTuner(ABC):
    "Abstract Class representing a BaseHyperTuner"
    dataset: Type[BaseData]
    directory: str
    project_name: str

    @abstractproperty
    def tuner(self):
        pass

    @abstractstaticmethod
    def build_model(hp):
        pass

    @abstractmethod
    def search(self):
        pass

    def get_model(self, trial_number: str):
        "Return the model for a trial number"
        trial = self.tuner.oracle.get_trial(trial_number)
        tf_model = self.tuner.load_model(trial)
        shape = (None, self.dataset.X_train.shape[1], self.dataset.X_train.shape[2])
        tf_model.build(input_shape=shape)
        return tf_model

    def dir(self, file_name: Optional[str] = ""):
        return os.path.join(self.directory, self.project_name, file_name)

    def save_hyper_tuner(self):
        "Save model to a cloudpickle file."
        save(self, self.dir("hyper_tuner.pickle"))

    def get_best_model(self, best_rank: int = 1):
        "Return a model by its ranking"
        best_model = self.tuner.get_best_models(num_models=best_rank)[best_rank - 1]
        shape = (None, self.dataset.X_train.shape[1], self.dataset.X_train.shape[2])
        best_model.build(input_shape=shape)
        return best_model
