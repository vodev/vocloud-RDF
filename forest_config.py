import random

class ForestConfig(object):
    """docstring for TreeConfig"""
    def __init__(self, json_data):
        super(ForestConfig, self).__init__()
        self.implementation = json_data['implementation']
        self.features =  json_data['features']
        self.label = json_data['label']

        self.trees = json_data['trees'] if 'trees' in json_data else 100
        self.min_split = json_data['min_split'] if 'min_split' in json_data else 4
        self.mode = json_data['mode'] if 'mode' in json_data else 'classification'
        self.splitting_criterion = json_data['splitting_criterion'] if 'splitting_criterion' in json_data else 'entropy'
        self.depth = json_data['depth'] if 'depth' in json_data else 1000
        self.sample_ratio = json_data['sample_ratio'] if 'sample_ratio' in json_data else 67
        self.bin_limit = json_data['bin_limit'] if 'bin_limit' in json_data else 1024
        self.seed = json_data['seed'] if 'seed' in json_data else int(random.random() * 1000000000)
        self.error_estimate = int(json_data['error_estimate']) if 'error_estimate' in json_data else True
        self.sampling = json_data['sampling'] if 'sampling' in json_data else 'RANDOM'
        self.threads = 1
        self.computing_nodes = []
        self.ignore = []
        if 'performance' in json_data:
            perf_conf = json_data['performance']
            self.threads = perf_conf['threads'] if 'threads' in perf_conf else -1
            self.computing_nodes = perf_conf['computing_nodes'] if 'computing_nodes' in perf_conf else []
        if 'ignore' in json_data:
            json_data_ignore = json_data['ignore']
            for column in json_data_ignore:
                self.ignore.append(str(column))
