import io
import os

import pandas as pd
import pyfits
from astropy.io.votable import parse
from astropy.io.votable.tree import VOTableFile, Resource, Table, Field
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.cross_validation import train_test_split


def load_header(uri):
    votable = parse(uri)
    table = votable.get_table_by_index(0)
    values =  table.array[0][1].tolist()
    return [str(value) for value in values] + ["class"]


def load_set(uri, format='csv', header=None, delimiter=','):
    print(str(header))
    return pd.read_csv(uri, header=None, names=header, sep=None, dtype=None,
                       na_values='?', skipinitialspace=True)


def _to_votable(data, file_name):
    votable = VOTableFile()
    resource = Resource()
    votable.resources.append(resource)
    table = Table(votable)
    resource.tables.append(table)
    columns = data.columns
    if data.columns[-1] == 'class':
        columns = columns[:-1]
    fields = [
        Field(votable, name="intensities", datatype="double", arraysize='*')]
    table.fields.extend(fields)
    table.create_arrays(1)
    table.array[0] = columns.tolist()
    votable.to_xml(file_name)


def _write_csv(data, uri, header=None, separator=',', dtypes=None):
    with io.open(uri, 'w', encoding='utf-8') as out:
        if header is not None:
            _to_votable(header,
                        "meta.xml")

        for row in data:
            rec_num = 0
            for record in row:
                val = record
                if (dtypes is not None and 'int' in str(dtypes[rec_num])):
                    val = int(val)
                elif (dtypes is not None and 'float' in str(dtypes[rec_num])):
                    val = float(val)
                out.write(str(val))
                if (rec_num != len(row) - 1):
                    out.write(separator)
                rec_num += 1
            out.write('\n')


def _parse_fits(uri):
    fits = pyfits.open(uri, memmap=False)
    dat = fits[1].data
    fits.close()
    return dat.tolist()


def split_train_set(uri, label=-1, ratio=0.67, sep=',', header=None):
    header_num = None if not header else 0
    array = pd.read_csv(uri, delimiter=sep, skipinitialspace=True,
                        na_values=['?'])
    train, test = train_test_split(array.values, train_size=ratio)
    base_name, ext = os.path.splitext(os.path.basename(uri))
    directory = os.path.dirname(uri)
    train_name = directory + base_name + '_train' + ext
    test_name = directory + base_name + '_score' + ext
    _write_csv(train, train_name, separator=sep, header=None)
    _write_csv(test, test_name, separator=sep, header=None)
    return (train_name, test_name)


def create_xvalidation_files(data_uri, data_conf, header, configuration,
                             target=None, base_folder='./result/xvalidation'):
    df = pd.read_csv(data_uri, sep=None, header=None, names=header,
                     skipinitialspace=True, na_values=['?'])
    kfold = None
    labels = df[target].values
    folds = configuration['folds']
    if (target is not None):
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
        _write_csv(df.values[train], train_uri, header=header, dtypes=df.dtypes)
        _write_csv(df.values[test], test_uri, header=header, dtypes=df.dtypes)
        uris.append((train_uri, test_uri))
        i += 1
    return uris
