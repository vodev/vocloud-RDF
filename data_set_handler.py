import pyfits
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold, KFold
import os
import io
from pprint import pprint

def process_set(uris, name, format='csv', normalize=True, binning=True, delimiter=','):
    if(format == 'csv'):
        return uris
    elif(format == 'fits'):
        all_fits = _parse_all_fits(uris)
        if(binning):
            all_fits = _binning(all_fits)
        if normalize:
            _normalize(all_fits)
        csv_name = 'processed_' + name + '.csv'
        _write_fits_csv(all_fits, csv_name)
        return csv_name

def load_set(uri, format='csv', header=False, delimiter=','):
    print(str(header))
    return pd.read_csv(uri, header=0 if header else None, sep=None, dtype=None, na_values='?', skipinitialspace=True)


def _binning(fits_list):
    '''do data binning'''
    result = []
    #pprint(fits_list)
    #pprint(fits_list[0])
    #pprint(fits_list[0]['data'][0])

    first_min = fits_list[0]['data'][0][0]
    first_max = fits_list[0]['data'][0][0]
    last_min = fits_list[0]['data'][-1][0]
    last_max = fits_list[0]['data'][-1][0]

    for fits in fits_list:
        first_min = min(fits['data'][0][0], first_min)
        first_max = max(fits['data'][0][0], first_max)
        last_min = min(fits['data'][-1][0], last_min)
        last_max = max(fits['data'][-1][0], last_max)
    first_avg = first_max
    last_avg = last_min
    diff = 0.25
    #print((first_min + first_max) / 2, (last_min + last_max) / 2)
    for fits in fits_list:
        fits_data = fits['data']
        binned_data = []
        current_val = first_avg
        it = 0
        columns = 0
        while current_val <= last_avg:
            while fits_data[it][0] > current_val or fits_data[it + 1][0] < current_val:
                it += 1
            diff_x = fits_data[it + 1][0] - fits_data[it][0]
            diff_y = fits_data[it][1] - fits_data[it + 1][1]
            diff_x_val = current_val - fits_data[it][0]
            div = diff_x_val / diff_x
            binned_data.append((current_val, fits_data[it][1] - diff_y * div))
            current_val += diff
            columns += 1
        binned_dictionary = {}
        binned_dictionary['data'] = binned_data
        binned_dictionary['id'] = fits['id']
        binned_dictionary['class'] = fits['class']
        result.append(binned_dictionary)
    return result


def _normalize(fits_list):
    '''normalize data'''


def _write_fits_csv(fits_list, name):
    csv_file = io.open(name, mode='w', encoding="utf-8")
    csv_file.write('id,')
    for record in fits_list[0]['data']:
        csv_file.write(str(record[0]))
        csv_file.write(',')
    csv_file.write('class\n')
    for fits in fits_list:
        #print(fits)
        csv_file.write(fits['id'])
        csv_file.write(',')
        for record in fits['data']:
            #print(str(record[1]))
            csv_file.write(str(record[1]))
            csv_file.write(',')
        csv_file.write(fits['class'])
        csv_file.write('\n')
    csv_file.close()


def _parse_all_fits(uri):
    parsed_fits = []
    classes = None
    current_class = None
    features = 1997
    for root, dirs, files in os.walk(uri):
        base = os.path.basename(root)
        #print(base)
        if root == uri:
            classes = dirs
        elif base in classes:
            current_class = base
        for fi in files:
            if(fi.endswith('.fits')):
                fits_data = _parse_fits(os.path.join(root, fi))
                if(len(fits_data) != features):
                    continue
                fits = {}
                fits['data'] = fits_data
                fits['id'] = fi
                fits['class'] = current_class
                #pprint.pprint(fits[-1])

                parsed_fits.append(fits)

    #pprint.pprint(parsed_fits)
    return parsed_fits


def _write_csv(data, uri, header=None, separator=',', dtypes=None):
    with io.open(uri, 'w', encoding='utf-8') as out:
        if header is not None or False:
            print("writing header")
            for record in header:
                try:
                    out.write(str(record))
                except TypeError:
                    out.write(unicode(str(record)))
                if(record != header[-1]):
                    out.write(separator)
            out.write('\n')

        for row in data:
            rec_num = 0
            for record in row:
                val = record
                if(dtypes is not None and 'int' in str(dtypes[rec_num])):
                    val = int(val)
                elif(dtypes is not None and 'float' in str(dtypes[rec_num])):
                    val = float(val)
                out.write(str(val))
                if(rec_num != len(row) - 1):
                    out.write(separator)
                rec_num += 1
            out.write('\n')


def _parse_fits(uri):
    fits = pyfits.open(uri, memmap=False)
    dat = fits[1].data
    fits.close()
    return dat.tolist()


def split_train_set(uri, label=-1, ratio=0.67, sep=',', header=True):
    header_num = None if not header else 0
    array = pd.read_csv(uri, header=header_num, delimiter=sep, skipinitialspace=True, na_values=['?'])
    train, test = train_test_split(array.values, train_size=ratio)
    base_name, ext = os.path.splitext(os.path.basename(uri))
    directory = os.path.dirname(uri)
    train_name = directory + base_name + '_train' + ext
    test_name = directory + base_name + '_score' + ext
    _write_csv(train, train_name, separator=sep, header=None if not header else array.columns)
    _write_csv(test, test_name, separator=sep, header=None if not header else array.columns)
    return (train_name, test_name)

def create_xvalidation_files(data_uri, data_conf, configuration, target = None, base_folder='./result/xvalidation'):
    has_header = data_conf['header'] if 'header' in data_conf else False
    header = 0 if has_header else None
    df = pd.read_csv(data_uri, sep=None, header=header, skipinitialspace=True, na_values=['?'])
    kfold = None
    labels = df[target].values
    folds = configuration['folds']
    if(target is not None):
        kfold = StratifiedKFold(labels, folds)
    else:
        kfold = KFold(n=len(labels), n_folds=folds)

    if not os.path.exists(base_folder):
        os.makedirs(os.path.abspath(base_folder))
    i = 1
    uris = []
    for train, test in kfold:
        train_uri = base_folder + '/train_' + str(i)
        test_uri = base_folder + '/test_' + str(i)
        _write_csv(df.values[train], train_uri, header=df.columns if has_header else None, dtypes=df.dtypes)
        _write_csv(df.values[test], test_uri, header=df.columns if has_header else None, dtypes=df.dtypes)
        uris.append((train_uri, test_uri))
        i += 1
    return uris

