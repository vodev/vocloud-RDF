import string
import json
import os
__DIV_TEMPLATE = string.Template('<div id="conf_matrix_${key}" style="height: 400px; min-width: 310px; max-width: 800px; margin: 0 auto"></div>\n')
__LINK_TEMPLATE = string.Template('<li><a id="${spectrum_name}_link" href="./spectra/spectrum_${spectrum_name}.html">${spectrum_name_short} as ${class}</a></li>\n')
__LINK_SECTION_TEMPLATE = string.Template('<section id="${key"}><h2>${key}</h2>\n${spectra_links}</section>')
__script_dir = os.path.dirname(os.path.realpath(__file__))
def output_to_html(data, data_sets=None, out_dir='./'):
    substitutes = {'f1_score': '', 'cat_names': '', 'conf_matrix_code':'', 'conf_matrix_div':'', 'n_trees':'', 'score':'', 'classified_data':''}
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
            conf_matrix_data[key] = __transform_conf_matrix_data(val['conf_matrix'])
            div_template += __DIV_TEMPLATE.substitute({'key':key})
        if 'predicted' in val:
            classified_data[key] = val['predicted']['test']
    substitutes['conf_matrix_div'] = __generate_conf_matrix_div(data.keys())
    substitutes['f1_score'] = json.dumps(f1_data)
    substitutes['cat_names'] = ['A', 'B', 'C', 'D']
    substitutes['conf_matrix_code'] = __generate_conf_matrix_code(conf_matrix_data, substitutes['cat_names'])
    substitutes['classified_data'], substitutes['classified_data_links'] = __generate_classified_data(classified_data, data_sets, out_dir)
    with open(out_dir +'/index.html', 'w') as out_file:
        out_file.write(html_template.substitute(substitutes))

def __transform_conf_matrix_data(data):
    out_data = []
    for idx, x_val in enumerate(data):
        for idx_2, y_val in enumerate(x_val):
            out_data.append([idx, idx_2, y_val])
    print('saving ' + str(out_data))
    return out_data

def __generate_classified_data(data, data_sets, out_dir):
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
        all_spectra = data_sets[key]['test_set'].values.tolist()
        categories = json.dumps([i for i in range(0, len(all_spectra[0]))])

        for index, value in enumerate(predicted):
            points = json.dumps(all_spectra[index])
            html_code = html_code_template.substitute({'name':key, 'points':points, 'cats':categories, 'class': predicted[index]})
            spectrum_name = key + '_' + str(index)
            link_code = __LINK_TEMPLATE.substitute({'class': predicted[index], 'spectrum_name':spectrum_name, 'spectrum_name_short': index})
            links += link_code
            with open(out_dir + 'spectrum_' + spectrum_name + '.html', 'w') as spectra_file:
                spectra_file.write(html_code)
    return pages, ''.join(links)

def __generate_conf_matrix_div(keys):
    '''Generates confusion matrix html divs for the keys'''
    divs = []
    for key in keys:
        divs.append(__DIV_TEMPLATE.substitute({'key':key}))
    return ''.join(divs)

def __generate_conf_matrix_code(data, cat_names):
    '''Generates javascript code out of provided data'''
    code_template = ''
    with open(__script_dir + '/conf_matrix_code.js.template') as template_file:
        code_template = string.Template(template_file.read())
    conf_matrix_codes = []
    for key, val in data.items():
        substitutes = {'key':key, 'conf_matrix_data':val, 'cat_names': cat_names}
        conf_matrix_codes.append(code_template.substitute(substitutes))
    return ''.join(conf_matrix_codes)

