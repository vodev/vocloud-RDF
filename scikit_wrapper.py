import sklearn.ensemble as ensemble
import sklearn.cross_validation as xv
import sklearn.preprocessing as prep
import sklearn.pipeline as pip
import sklearn.metrics as met
try:
    import hybridforest
except ImportError:
    pass
import sklearn as sk
import pandas as pd
import base_wrapper
import os
import numpy as np
from pprint import pprint
class SkLearnForest(base_wrapper.BaseForest):
    """docstring for SkLearnForest"""
    def __init__(self, wrapper, forest_config, instance):
        super(SkLearnForest, self).__init__(wrapper, forest_config)
        self.instance = instance

    def predict(self, data):
        data_set = self.wrapper.data[data]
        columns = [column for column in data_set.columns if column not in self.forest_config.ignore and column != self.forest_config.label]
        features = data_set[columns].values
        return self.instance.predict(X=features)

    def get_progress(self):
        return 1.0

    def score(self, data=None):
        data_set = self.wrapper.data[data]
        target = data_set[self.forest_config.label].values
        columns = [column for column in data_set.columns if column not in self.forest_config.ignore and column != self.forest_config.label]
        features = data_set[columns].values
        pprint("scoring on " + str(self.instance))
        return self.instance.score(features, target)

    def oob_score(self):
        try:
            return self.forest_config.trees, self.instance.oob_score_
        except AttributeError:
            return self.forest_config.trees, None

    def f1_score(self, data_key):
        predicted = self.predict(data_key)
        data = self.wrapper.data[data_key]
        return met.f1_score(data[self.forest_config.label], predicted, average=None).tolist()

    def confusion_matrix(self, data_key):
        predicted = self.predict(data_key)
        data = self.wrapper.data[data_key]
        return met.confusion_matrix(data[self.forest_config.label], predicted).tolist()


class SkLearnWrapper(base_wrapper.BaseWrapper):
    """docstring for SkLearnWrapper"""
    data = {}
    def __init__(self, config):
        super(SkLearnWrapper, self).__init__()
        self.config = config

    def __create_sk_instance(self, config):
        if(config.implementation == 'scikit' or config.implementation == 'all'):
            if config.mode == 'classification':
                instance = ensemble.RandomForestClassifier(n_estimators=config.trees, 
                                                     criterion=config.splitting_criterion.lower(),
                                                     max_features=config.features,
                                                     min_samples_split=config.min_split,
                                                     n_jobs=config.threads, random_state=config.seed,
                                                     verbose=0, oob_score=True)
            elif  config.mode == 'gpu':
                instance = hybridforest.RandomForestClassifier(n_estimators=config.trees, 
                                                     max_features=config.features,
                                                     bootstrap=True, n_jobs=config.threads)
            else:
                instance = ensemble.RandomForestRegressor(n_estimators=config.trees,
                                                     max_features=config.features,
                                                     min_samples_split=config.min_split,
                                                     n_jobs=config.threads, random_state=config.seed,
                                                     verbose=0, oob_score=True)
            return instance

    def import_data(self, uri, header=0, config=None):
        key = os.path.basename(uri)
        print("header " + str(header))
        if key not in SkLearnWrapper.data:
            SkLearnWrapper.data[key] = pd.read_csv(uri, header=0 if header else None, sep=None, dtype=None, na_values='?', skipinitialspace=True)
        return key

    def train_forest(self, data_key):
        data_set = SkLearnWrapper.data[data_key]
        pprint(data_set)
        target = data_set[self.config.label].values
        pprint(target)
        columns = [column for column in data_set.columns if column not in self.config.ignore and column != self.config.label]
        features = data_set[columns].values
        pprint(features)
        pprint(data_set.dtypes)
        instance = self.__create_sk_instance(self.config)
        pipe = pip.Pipeline([("imputer", prep.Imputer(
                                          strategy="most_frequent",
                                          axis=0, verbose=50000)),
                      ("forest", instance)])

        pipe.fit(features, target)
        return SkLearnForest(self, self.config, instance)

    def xvalidation(self, folds, data_key):
        if(self.config.mode == 'regression'):
            raise TypeError()
        data_set = SkLearnWrapper.data[data_key]
        target = data_set[self.config.label].values
        columns = [column for column in data_set.columns if column not in self.config.ignore and column != self.config.label]
        features = data_set[columns].values
        instance = self.__create_sk_instance(self.config)
        pipe = pip.Pipeline([("imputer", prep.Imputer(
                                          strategy="mean",
                                          axis=0)),
                      ("forest", instance)])
        return xv.cross_val_score(pipe, cv=folds, scoring='accuracy', X=features, y=target)
