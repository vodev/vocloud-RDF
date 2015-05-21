from subprocess import Popen, STDOUT, PIPE
from threading import Thread, Condition
import sys
import pdb
import requests
import time
from re import search
from pprint import pprint
import numpy
from base_wrapper import BaseForest, BaseWrapper
try:
    from subprocess import DEVNULL # py3k
except ImportError:
    import os
    DEVNULL = open(os.devnull, 'wb')

#http://localhost:54321/RFView.html?data_key=final.hex&model_key=__RFModel__afd38bcf7d1b9197dd5c28c6fe83431b&response_variable=class&ntree=100&class_weights=1%3D1.0%2C2%3D1.0%2C3%3D1.0%2C4%3D1.0&no_confusion_matrix=0&clear_confusion_matrix=1&iterative_cm=1&refresh_threshold_cm=1
def _transform_args(args):
    for key, val in args.items():
        if(isinstance(val, bool)):
            args[key] = int(val)
       
class H2OForest(BaseForest):
    """docstring for H2OForest"""

    RF_VIEW = 'RFView'
    RF_SCORE = 'RFScore'
    PREDICT = 'Predict'
    def __init__(self, wrapper, forest_config, train_args):
        super(H2OForest, self).__init__(wrapper, forest_config)
        self.wrapper = wrapper
        self.forest_config = forest_config
        self.__train_args = train_args
        self.number_of_trees = self.__train_args['ntree']
        self.key = train_args['model_key']

    def get_progress(self, args=None):
        request_args = self.__train_args
        #request_args["out_of_bag_error_estimate"] = 0
        request_args["iterative_cm"] = 0
        if(args is not None):
            request_args = args
        data = self.wrapper.request(H2OForest.RF_VIEW, request_args)
        return 1.0 if data['response']['status'] == 'done' else data['response']['progress'] / data['response']['progress_total']


    def predict(self, data):
        if not self.finished():
            raise Exception("Not trained")
        args = {'model':self.key, 'data':data}
        data = wrapper.request(PREDICT, args)
        redirect_uri = data['response']['redirect_request']
        redirect_args = data['response']['redirect_request_args']
        while(data['response']['status'] != 'done'):
            args = {'model':self.key, 'data':data}
            data = wrapper.request(redirect_uri, redirect_args)


    def score(self, data):
        args = {'model_key': self.key, 'data_key': data}
        data = self.wrapper.request(H2OForest.RF_SCORE, args)
        redirect_uri = data['response']['redirect_request']
        redirect_args = data['response']['redirect_request_args']
        
        args = {'model_key': redirect_args['model_key'],
                'destination_key': redirect_args['destination_key'],
                'data_key': redirect_args['data_key']} 
        while(not self.get_progress(args=args)):
            time.sleep(1)     
        data = self.wrapper.request(redirect_uri, args)
        return 1 - data['confusion_matrix']['classification_error']

    def training_con_matrix(self):
        args = {'model_key': self.key, 'refresh_threshold_cm': 1, 'data_key':
                self.__train_args['data_key'], 'out_of_bag_error_estimate': 1}
        data = self.wrapper.request(H2OForest.RF_VIEW, args)
        if('confusion_matrix' in data and 'classes_errors' in data['confusion_matrix']):
            if(self.finished()):
                return data['ntree'], data['confusion_matrix']['classes_errors']
            else:
                return data['response']['progress'], data['confusion_matrix']['classes_errors']

        else:
            return data['response']['progress'], None

    def oob_score(self):
        args = {'model_key': self.key, 'refresh_threshold_cm': 1, 'data_key':
                self.__train_args['data_key'], 'out_of_bag_error_estimate': 1}
        data = self.wrapper.request(H2OForest.RF_VIEW, args)
        if('confusion_matrix' in data and 'classes_errors' in data['confusion_matrix']):
            if(self.finished()):
                return data['ntree'], 1 - data['confusion_matrix']['classification_error']
            else:
                return data['response']['progress'], 1 - data['confusion_matrix']['classification_error']

        else:
            return data['response']['progress'], None


class H2OWrapper(object):
    """docstring for H2OWrapper"""
    IMPORT_FILE = 'ImportFiles'
    IMPORT_URL = 'ImportUrl'
    PARSE = 'Parse'
    BUILD_TREE = 'RF'
    TO_ENUM='ToEnum'
    #PREDICT = '/'
    __imported_files = {}

    def __init__(self, forest_config):
        super(H2OWrapper, self).__init__()
        self.forest_config = forest_config
        self.__host = 'http://localhost:54321/'
        self.__lock = Condition()
        self.__process = None
        r = None
        try:
            r = requests.get(self.__host)
        except:
            self.__start_h2o()


    def request(self, uri='', args=None):
        url = self.__host + uri
        if(not args is None):
            _transform_args(args)
        data = requests.get(url, params=args)
        #pprint(data.url)
        #pprint(data.json())
        return data.json()

    def read_ip(self, out):
        #print('Listening for HTTP and REST traffic on  http://192.168.1.98:54321/')
        regex = 'Listening for HTTP and REST traffic on  (http://[a-zA-Z0-9.]*:[0-9]*/)$'
        #pdb.set_trace()
        #print("listening")
        for line in iter(lambda: out.readline().decode('utf-8'), b''):
            #pprint(str(line))
            #print('Line is ' + line)
            match = search(regex, line)
            if(match is not None):
                #print('matched line ' + line)
                #pprint('found match ' + match.group(1))
                self.__host = match.group(1)
                return
        with self.__lock:
            self.__lock.notify()

    def __start_h2o(self):
        ON_POSIX = 'posix' in sys.builtin_module_names

        self.__process = Popen('java -jar h2o.jar', stdout=PIPE, bufsize=1, close_fds=ON_POSIX, shell=True)
        self.read_ip(self.__process.stdout)
        self.__process.stdout = DEVNULL
        #self.__read_ip(p.stdout)
        #time.sleep(60)

    def __import_file(self, uri, header):
        args = {'path': uri}
        data = self.request(H2OWrapper.IMPORT_FILE, args)
        key = data['keys'][0]
        args = {'source_key': key, 'header': header}
        data = self.request(H2OWrapper.PARSE, args)
        key = data['destination_key']
        redirect = data['response']['redirect_request']
        args = data['response']['redirect_request_args']
        data = self.request(redirect, args)['response']
        while (data['status'] != 'redirect'):
            data = self.request(redirect, args)['response']
            #redirect = data['redirect_request']
            #args = data['redirect_request_args']
        #time.sleep(60)
        #time.sleep(5)
        self.request(H2OWrapper.TO_ENUM, {'key':key,'col_index':self.forest_config.label,'to_enum':1})
        print('importing finished')
        return key

    def import_data(self, uri, header=False, config=None):
        '''import file'''
        if(uri not in H2OWrapper.__imported_files):
            H2OWrapper.__imported_files[uri] = self.__import_file(uri, header=1 if header else 0)
        return H2OWrapper.__imported_files[uri]
        

    def train_forest(self, data_key):
        '''train forest'''
        #http://192.168.1.98:54321/2/DRF.query?destination_key=&source=final.hex&response=class&ignored_cols=0&classification=1&validation=&ntrees=50&max_depth=20&min_rows=4&nbins=1600&score_each_iteration=0&importance=1&mtries=-1&sample_rate=0.6666666865348816&seed=-1&build_tree_one_node=1
        ignore = ','.join(self.forest_config.ignore)
        args = {'data_key': data_key, 'ntree': self.forest_config.trees, 
        'features': self.forest_config.features, 'depth': self.forest_config.depth, 
        'stat_type': self.forest_config.splitting_criterion, 'sample': self.forest_config.sample_ratio,
        'out_of_bag_error_estimate': 1,
        'bin_limit':self.forest_config.bin_limit, 'seed':self.forest_config.seed, 
        'parallel': 1, 'ignore': ignore, 'response_variable': self.forest_config.label, 'iterative_cm':0}
        #args = {}
        data = self.request(H2OWrapper.BUILD_TREE, args)
        pprint(data)

        key = data['destination_key']
        redirect = data['response']['redirect_request']
        args = data['response']['redirect_request_args']
        #data = self.request(redirect, args)
        forest = H2OForest(self, self.forest_config, args)
        while(not forest.finished()):
            time.sleep(1)
        return forest
