import string
import json
import os
import pandas as pd
from pprint import  pprint
__DIV_TEMPLATE = string.Template(
    '<div id="conf_matrix_${key}" style="height: 400px; min-width: 310px; max-width: 800px; margin: 0 auto"></div>\n')
__LINK_TEMPLATE = string.Template(
    '<option id="${spectrum_name}_link">${spectrum_name}</option>\n')
__LINK_SECTION_TEMPLATE = string.Template('<section id="${key"}><h2>${key}</h2>\n${spectra_links}</section>')
__script_dir = os.path.dirname(os.path.realpath(__file__))


def __group_spectra(logged_data, data_sets):
    groups = {}
    for key, logged_data_for_key in logged_data.items():
        matrix_labels = logged_data_for_key["conf_matrix"]["data"][0]
        score_set = data_sets[key]["score_set"]
        groups[key] = {}
        for idx, label in enumerate(matrix_labels):
            spectrum = score_set.iloc[idx]
            true_label = spectrum['class']
            coordinates = true_label, label
            if coordinates not in groups[key]:
                groups[key][coordinates] = pd.DataFrame()
            groups[key][coordinates] = groups[key][coordinates].append(spectrum)

    return groups


def _generate_conf_matrix_subpage(key, coordinate, spectra):
    with open(__script_dir + "/spectra_list.html.template") as template_file:
        html_template = string.Template(template_file.read())
    spectra_list = []
    for index, spectrum in spectra.iterrows():
        spectrum_link = __LINK_TEMPLATE.substitute({'class': coordinate[1], 'spectrum_name': key + "_" + spectrum["id"],
                                                    'spectrum_name_short': spectrum["id"]})
        spectra_list.append(spectrum_link)
    categories = json.dumps([float(item) for item in spectra.columns.values.tolist()[:-2]])

    html_code = html_template.substitute(
        {"list": "\n".join(spectra_list), "true_class": coordinate[0], "classified_class": coordinate[1],
         "cats": categories})
    return html_code


def __generate_conf_matrix_pages(logged_data, data_sets, out_dir):
    all_groups = __group_spectra(logged_data, data_sets)
    """ out_dir += "/matrix_subpages/"
    try:
        os.mkdir(out_dir)
    except OSError:
        pass"""
    for key, groups in all_groups.items():
        for coordinate, spectra in groups.items():
            code = _generate_conf_matrix_subpage(key, coordinate, spectra)
            # we have "normalize" classes so they will be integers starting from 0"
            classes = sorted(data_sets[key]["score_set"]["class"].unique())
            normalized_classes = {val: idx for idx, val in enumerate(classes)}
            matrix_dir = out_dir + "/" + str(normalized_classes[coordinate[0]]) + "_" \
                         + str(normalized_classes[coordinate[1]])
            try:
                os.mkdir(matrix_dir)
            except OSError:
                pass
            with open(matrix_dir + "/matrix.html", "w") as out_file:
                out_file.write(code)
            __generate_spectra(logged_data, spectra, matrix_dir)


def __generate_spectra(logger_data, group, out_dir):
    group_without_id_and_class = group.drop(["id", "class"], axis=1)
    print("saving spectra to " + out_dir)
    group_without_id_and_class.to_csv(path_or_buf=out_dir + "/spectra.txt", sep=",", header=False, index=False)


def output_to_html(data, data_sets=None, out_dir='./'):
    """Transforms the logged data to an html output.
    Arguments: data - dictionary that will be outputted
               data_sets - collection of all the datasets used in this run. Generally has the following structur:
                data_sets[run_key][data_set_type] where data_set_type may be train_set, test_set or score_set"""
    substitutes = {'f1_score': '', 'cat_names': '', 'conf_matrix_code': '', 'conf_matrix_div': '', 'n_trees': '',
                   'score': '', 'classified_data': ''}
    html_template = ''
    div_template = ''
    with open(__script_dir + '/index.html.template') as template:
        html_template = string.Template(template.read())
    f1_data = []
    conf_matrix_data = {}
    conf_matrx_js_strings = []
    keys = data.keys()
    classified_data = {}
    for key, val in data.items():
        if 'f1_score' in val:
            f1_data.append({'name': key, 'data': val['f1_score']})
        if 'conf_matrix' in val:
            conf_matrix_data[key] = __transform_conf_matrix_data(val['conf_matrix']['matrix'])
            div_template += __DIV_TEMPLATE.substitute({'key': key})
            __generate_conf_matrix_pages(data, data_sets, out_dir)
        if 'predicted' in val:
            classified_data[key] = val['predicted']['test']
    # __generate_conf_matrix_pages(val['conf_matrix']['data'])

    substitutes['conf_matrix_div'] = __generate_conf_matrix_div(data.keys())
    substitutes['f1_score'] = json.dumps(f1_data)
    substitutes['cat_names'] = sorted(data_sets[list(data_sets.keys())[0]]["train_set"]["class"].unique())
    substitutes['conf_matrix_code'] = __generate_conf_matrix_code(conf_matrix_data, substitutes['cat_names'])
    substitutes['classified_data'], substitutes['classified_data_links'] = __generate_data(classified_data, data_sets,
                                                                                           "test_set", out_dir)
    with open(out_dir + '/index.html', 'w') as out_file:
        out_file.write(html_template.substitute(substitutes))


def __transform_conf_matrix_data(data):
    out_data = []
    for idx, x_val in enumerate(data):
        for idx_2, y_val in enumerate(x_val):
            out_data.append([idx, idx_2, y_val])
    print('saving ' + str(out_data))
    return out_data


def __generate_data(data, data_sets, set_key, out_dir):
    '''generates links to spectra plots and also creates pages for all spectras'''
    out_dir += '/spectra/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    html_code_template = ''
    links = []
    pages = {}
    with open(__script_dir + '/spectra_plot.html.template') as template_file:
        html_code_template = string.Template(template_file.read())
    res = ''
    for key, predicted in data.items():
        pages[key] = []
        all_spectra = data_sets[key][set_key]
        categories = json.dumps([i for i in range(0, len(all_spectra[0]))])

        for index, value in enumerate(predicted):
            if "class" in all_spectra:
                points = json.dumps(all_spectra.iloc[index].tolist()[0:-1])
            else:
                points = json.dumps(all_spectra.iloc[index].tolist())
            html_code = html_code_template.substitute({'name': key, 'points': points, 'cats': categories,
                                                       'class': predicted[index]})
            spectrum_name = key + '_' + all_spectra.iloc[index]["id"]
            link_code = __LINK_TEMPLATE.substitute({'class': predicted[index], 'spectrum_name': spectrum_name,
                                                    'spectrum_name_short': index})
            links += link_code
            with open(out_dir + 'spectrum_' + spectrum_name + '.html', 'w') as spectra_file:
                spectra_file.write(html_code)
    return pages, ''.join(links)


def __generate_conf_matrix_div(keys):
    '''Generates confusion matrix html divs for the keys'''
    divs = []
    for key in keys:
        divs.append(__DIV_TEMPLATE.substitute({'key': key}))
    return ''.join(divs)


def __generate_conf_matrix_code(data, cat_names):
    '''Generates javascript code out of provided data'''
    code_template = ''
    with open(__script_dir + '/conf_matrix_code.js.template') as template_file:
        code_template = string.Template(template_file.read())
    conf_matrix_codes = []
    for key, val in data.items():
        substitutes = {'key': key, 'conf_matrix_data': val, 'cat_names': cat_names}
        conf_matrix_codes.append(code_template.substitute(substitutes))
    return ''.join(conf_matrix_codes)
