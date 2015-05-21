import json
import io
import os
import string
class ResultLog(object):
    """docstring for ResultLog"""
    out_data = {}
    output_dir = os.path.abspath('./results')
    def __init__(self, key_prefix=None):
        super(ResultLog, self).__init__()
        print("Creating logger with prefix " + key_prefix)
        self.key_prefix = key_prefix.replace('.json', '')
        if(self.key_prefix not in ResultLog.out_data):
            print('adding ' + self.key_prefix + ' to out_data')
            ResultLog.out_data[self.key_prefix] = {}
    def add_result(self, key, value):
        print('adding result  with prefix ' + self.key_prefix + ' ' + key + ':' + str(value))
        keys = key.split('/')
        out_data = ResultLog.out_data[self.key_prefix]
        for key_part in keys:
            if key_part == keys[-1]:
                if(not key_part in out_data):
                    out_data[key_part] = []
                out_data[key_part].append(value)
            else:
                if(not key_part in out_data):
                    out_data[key_part] = {}
                out_data = out_data[key_part]
    
    @classmethod
    def write_result(cls, result_file):
        json_data = json.dumps(cls.out_data)
        out = io.open(cls.output_dir + '/' + result_file, 'w', encoding='utf-8')
        try:
            out.write(json_data)
        except TypeError:
            unicode_data = unicode(json_data)
            out.write(unicode_data)

    @classmethod
    def set_output_dir(cls, dirname):
        cls.output_dir = os.path.abspath(dirname)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
