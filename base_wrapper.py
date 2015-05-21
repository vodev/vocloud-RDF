from abc import ABCMeta, abstractmethod
import numpy

class BaseForest(object):
    __metaclass__ = ABCMeta

    def __init__(self, wrapper, forest_config):
        self.wrapper = wrapper
        self.forest_config = forest_config

    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def get_progress(self):
        pass

    def finished(self):
        return numpy.allclose(self.get_progress(), 1.0)

    @abstractmethod
    def oob_score(self):
        pass

    @abstractmethod
    def score(self, data=None):
        pass


class BaseWrapper(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def import_data(self, uri, header=0, config=None):
        pass

    @abstractmethod
    def train_forest(self, data_key):
        pass
