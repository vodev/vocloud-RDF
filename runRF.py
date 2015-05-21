import json
import sys
import h2o
import forest_config
import data_set_handler
import scikit_wrapper
import numpy as np
import result_log
import html_output
from datetime import datetime

BIN_FOLDER = 'bin/'
DEPENDENCIES_FOLDER = 'dependencies/'

IMPLEMENTATIONS = {'h2o': h2o.H2OWrapper,
                   'scikit': scikit_wrapper.SkLearnWrapper}

__data_sets = {}

def compute_conf_matrix(forest, test_set_uri, test_set_config, logger):
    '''Computes confusion matrix.
    Input is a forest object,
    uri to the test set
    and its config and and a logger'''
    data_key = forest.wrapper.import_data(test_set_uri,
                                          header=False if
                                          test_set_config is None
                                          else test_set_config['header'])
    matrix = forest.confusion_matrix(data_key)
    for value in matrix:
        logger.add_result('conf_matrix', value)
    return matrix

def compute_f1_score(forest, test_set_uri, test_set_config, logger):
    data_key = forest.wrapper.import_data(test_set_uri, header=False if test_set_config is None else test_set_config['header'])
    score = forest.f1_score(data_key)
    for value in score:
        logger.add_result('f1_score', value)
    return score


def run_induction(implementation, train_set_uri, train_set_config, logger):
    start = datetime.utcnow()
    key = implementation.import_data(train_set_uri,
                                     header=train_set_config['header'])
    forest = implementation.train_forest(key)
    end = datetime.utcnow()
    trees, matrix = forest.oob_score()
    logger.add_result('training_time', (trees, (end - start).total_seconds()))

    logger.add_result('training_oob_scores', (trees, matrix))
    return forest

def run_score(forest, test_set_uri, test_set_config, logger, header):
    test_key = forest.wrapper.import_data(test_set_uri,
                                          header=header)
    scores = forest.score(test_key)
    logger.add_result('score',
                     {'test_set_uri': (forest.forest_config.trees, scores)})
    return scores

def run_test(forest, test_set_uri, test_set_config, logger, header):
    test_key = forest.wrapper.import_data(test_set_uri, header=header)
    predicted_values = forest.predict(test_key).tolist()
    #logger.add_result('predicted', (test_set_uri, forest.forest_config.trees, predicted_values.tolist()))
    for score in predicted_values:
        logger.add_result('predicted/test', score)
    return predicted_values


def run_xvalidation(configuration, implementation, original_data_uri, train_set_config, logger,score_set_header, label=None):
    #check if the wrapper supports x-validation, if not, we will do our own
    n_folds = configuration['folds']
    print('%d fold xvalidation' % n_folds)
    forest = None
    try:
        xval = implementation.xvalidation
        key = implementation.import_data(original_data_uri, header=False if test_set_config is None else test_set_config['header'])
        result = implementation.xvalidation(n_folds, key)
    except(AttributeError, TypeError):
        if not run_xvalidation.files_generated:
            xvalidation_uris = data_set_handler.create_xvalidation_files(original_data_uri, 
                                train_set_config, configuration, target=label, base_folder='./xval')
            run_xvalidation.xvalidation_uris = xvalidation_uris
            run_xvalidation.files_generated = True
        xvalidation_uris = run_xvalidation.xvalidation_uris
        result = []
        for train_uri, test_uri in xvalidation_uris:
            forest = run_induction(implementation, train_uri, train_set_config, logger)
            result.append(run_score(forest, test_uri, train_set_config, logger, score_set_header))
    return np.mean(result)
    print('xvalidation presision: ' + str(np.mean(result)))


def run_wrapper(input_file):
    json_data = open(input_file)
    data = json.load(json_data)
    logger = result_log.ResultLog(input_file)
    result_log.ResultLog.set_output_dir(data['result'] if 'result' in data else '.')
    forest_conf = forest_config.ForestConfig(data)
    train_set = data['data_sets']['train_set']
    score_set = data['data_sets']['score_set'] if 'score_set' in data['data_sets'] else None
    test_set = data['data_sets']['test_set'] if 'test_set' in data['data_sets'] else None
    train_uri = data_set_handler.process_set(train_set['path'], data['name'],
                                             format=train_set['file_type'])
    test_uri = None
    score_uri = None


    if score_set is not None:
        score_uri = data_set_handler.process_set(score_set['path'], data['name'],
                                           format=score_set['file_type'])
    if test_set is not None:
        test_uri = data_set_handler.process_set(test_set['path'], data['name'],
                                           format=test_set['file_type'])

    elif 'split_dataset' in data and data['split_dataset']:
        train_uri, score_uri = data_set_handler.split_train_set(train_uri,
                                                                data['split_ratio'],
                                                                header=train_set['header'])
    score_set_header = train_set['header'] if score_set is None or score_set['header'] is None else score_set['header']
    test_set_header = train_set['header'] if test_set is None or test_set['header'] is None else test_set['header']
    print(train_uri + " " + score_uri)
    __data_sets[input_file.replace('.json','')] = {'train_set': data_set_handler.load_set(train_uri, header=train_set['header']),
                               'score_set': None if score_uri is None else data_set_handler.load_set(score_uri,
                                header=score_set_header),
                               'test_set': None if test_uri is None else data_set_handler.load_set(test_uri,
                                header=test_set_header)}

    implementation_cls = IMPLEMENTATIONS[forest_conf.implementation]
    implementation = implementation_cls(forest_conf)
    x_validation_params = data['x-validation'] if 'x-validation' in data else None
    if x_validation_params is not None:
        xvalidation_precision = run_xvalidation(x_validation_params, implementation, train_uri, train_set, logger, score_set_header, label=forest_conf.label)
        logger.add_result('xvalidation_precision', (forest_conf.trees, xvalidation_precision))

    forest = run_induction(implementation, train_uri, train_set, logger)
    if score_uri is not None:
        run_score(forest, score_uri, score_set, logger, score_set_header)
        if 'compute_f1' in data and data['compute_f1']:
            compute_f1_score(forest, score_uri, score_set, logger)
        if 'compute_cf' in data and data['compute_cf']:
            compute_conf_matrix(forest, score_uri, score_set, logger)
    if test_uri is not None:
        run_test(forest, test_uri, test_set, logger, test_set_header)
    print('Run for file ' + input_file + ' ended');


def main():
    run_xvalidation.files_generated = False
    if len(sys.argv) < 2:
        print('You need to specify at least one input file!')
        sys.exit(1)

    for arg in sys.argv[1:]:
        print('Running for input file ' + arg)
        run_wrapper(arg)

    result_log.ResultLog.write_result('result.json')

    html_output.output_to_html(result_log.ResultLog.out_data,
                               data_sets=__data_sets,
                               out_dir=result_log.ResultLog.output_dir)

if __name__ == '__main__':
    main()
