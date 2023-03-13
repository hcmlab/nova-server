
class TrainerClass:
    """Includes all the necessary files to run this script"""

    def __init__(self, ds_iter, logger, request_form=None):
        self.model = None
        self.ds_iter = ds_iter
        self.logger = logger
        self.data = None
        self.predictions = None
        self.DEPENDENCIES = []
        self.OPTIONS = {}
        self.request_form = request_form

    def preprocess(self):
        """Possible pre-processing of the data."""
        pass

    def train(self):
        """Trains a model with the given data."""
        pass

    def predict(self):
        """Predicts the given data with the given model. Stores a list with the predicted values to self.predictions."""
        pass

    def postprocess(self) -> list:
        """Possible pro-processing of the data. Returns a list with the pro-processed data."""
        return self.predictions

    def save(self, path) -> str:
        """Stores the weights of the given model at the given path. Returns the path of the weights."""
        modelpath = "pathtofile"
        return modelpath

    def load(self, path):
        """Loads a model with the given path. Returns this model."""
        pass
