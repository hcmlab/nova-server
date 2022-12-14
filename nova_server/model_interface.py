import logging

from hcai_datasets.hcai_nova_dynamic.hcai_nova_dynamic_iterable import HcaiNovaDynamicIterable


# TODO Check parameter types and correct them if necessary
class ModelInterface:
    """Includes all the necessary files to run this script"""
    DEPENDENCIES = []
    OPTIONS = {}

    def preprocess(self, ds_iter:  HcaiNovaDynamicIterable, logger: logging) -> list:
        """Possible pre-processing of the data. Returns a list with the pre-processed data."""
        pass

    def train(self, data_list: list, logger: logging) -> object:
        """Trains a model with the given data. Returns this model."""
        pass

    def predict(self, model, data_list: list, logger: logging) -> list:
        """Predicts the given data with the given model. Returns a list with the predicted values."""
        pass

    def postprocess(self, ds_iter: HcaiNovaDynamicIterable, logger: logging) -> list:
        """Possible pre-processing of the data. Returns a list with the pre-processed data."""
        pass

    def save(self, model, path: str, logger: logging) -> str:
        """Stores the weights of the given model at the given path. Returns the path of the weights."""
        pass

    def load(self, path: str, logger: logging) -> object:
        """Loads a model with the given path. Returns this model."""
        pass

