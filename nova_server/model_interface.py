
class ModelInterface:
    """Includes all the necessary files to run this script"""

    def __init__(self, ds_iter, logger):
        self.model = None
        self.ds_iter = None
        self.logger = None
        self.predictions = None
        self.DEPENDENCIES = []
        self.OPTIONS = {}

    def preprocess(self):
        """Possible pre-processing of the data. Returns a list with the pre-processed data."""
        pass

    def train(self):
        """Trains a model with the given data. Returns this model."""
        pass

    def predict(self):
        """Predicts the given data with the given model. Returns a list with the predicted values."""
        pass

    def postprocess(self) -> list:
        """Possible pro-processing of the data. Returns a list with the pro-processed data."""
        pass

    def save(self, path) -> str:
        """Stores the weights of the given model at the given path. Returns the path of the weights."""
        pass

    def load(self, path):
        """Loads a model with the given path. Returns this model."""
        pass
