import pickle
from enum import Enum

class Model(Enum):
    GCN = 1
    GAT = 2
    GSAGE = 3

class Dataset(Enum):
    CORA = 1
    CITESEER = 2
    PUBMED = 3

class AcceptFn(Enum):
    CONSTANT = 1
    INCREASING = 2

class SelectFn(Enum):
    BINARY = 1
    DECAYING = 2
    NONE = 3

class Reverse(Enum):
    REVERSED = 1
    ATTACKED = 2

class Recording:
    def __init__(self, losses, accuracies, iterations, model, dataset, accept, select, is_reserved):
        self.losses = losses
        self.accuracies = accuracies
        self.iterations = iterations
        self.model = model
        self.dataset = dataset
        self.accept = accept
        self.select = select
        self.is_reserved = is_reserved

    def save(self, filepath):
        """Save the current instance to a file."""
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filepath):
        """Load an instance of Recording from a file."""
        with open(filepath, 'rb') as file:
            return pickle.load(file)