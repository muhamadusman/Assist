import config
import numpy as np
import pandas as pd
from assist. experiment_names_lookup import diffusion_paper_313_datasets, diffusion_paper_2021_datasets

names_to_table_isbi = {'UNet-202209-Brats313-256-geoaug-imaug-valid-lr5.0e-02': '\\checkmark & 0 &',
                       'UNet-202209-Brats313-synthetic_1GAN-256-geoaug-imaug-valid-lr5.0e-02': '\\checkmark & 1 &',
                       'UNet-202209-Brats313-synthetic_5GANs-256-geoaug-imaug-valid-lr5.0e-02': '\\checkmark & 5 &',
                       'UNet-202209-Brats313-synthetic_10GANs-256-geoaug-imaug-valid-lr5.0e-02': '\\checkmark & 10 &',
                       'UNet-202209-Brats313-synthetic_20GANs-256-geoaug-imaug-valid-lr5.0e-02': '\\checkmark & 20 &',
                       'UNet-202209-synthetic_1GAN-256-geoaug-imaug-valid-lr5.0e-02': ' & 1 &',
                       'UNet-202209-synthetic_5GANs-256-geoaug-imaug-valid-lr5.0e-02': ' & 5 &',
                       'UNet-202209-synthetic_10GANs-256-geoaug-imaug-valid-lr5.0e-02': ' & 10 &',
                       'UNet-202209-synthetic_20GANs-256-geoaug-imaug-valid-lr5.0e-02': ' & 20 &'}


def get_names_to_table(base_dset_name: str, dset_to_table):
    # create names to table for diffusion paper by looking at diffusion_paper_313_datasets and
    # diffusion_paper_2021_datasets in experiment_names_lookup.py

    orig_augs = ((True, True), (True, False), (False, True), (False, False))
    setups = (' & \\checkmark & \\checkmark &', ' & \\checkmark & &', ' & &  \\checkmark &', ' & & &')

    names_to_table = {}
    for orig_aug, setup in zip(orig_augs, setups):
        start_str = 'UNet-202209-'
        if orig_aug[0]:
            start_str += base_dset_name + '-'

        end_str = '-256'
        if orig_aug[1]:
            end_str += '-geoaug-imaug'

        if orig_aug[0]:
            names_to_table[start_str + end_str[1:]] = 'None' + setup

        for name, entry in dset_to_table.items():
            names_to_table[start_str + name + end_str] = entry + setup

    return names_to_table


names_to_table_diffusion_paper_313 = get_names_to_table('Brats313', diffusion_paper_313_datasets)
names_to_table_diffusion_paper_2021 = get_names_to_table('Brats2021Train', diffusion_paper_2021_datasets)


def print_table(table: str):
    sheets = ('gd_enhancing_tumor', 'peritumoral_edema', 'necrotic_non_enhancing_tumor_co', 'mean')
    metric = 'dice'

    if table == 'diffusion_paper_313':
        expnames_to_table = names_to_table_diffusion_paper_313
        summary_file = config.results / 'model_evaluation' / 'summary_for_diffusion_paper313.xlsx'
    elif table == 'diffusion_paper_2021':
        expnames_to_table = names_to_table_diffusion_paper_2021
        summary_file = config.results / 'model_evaluation' / 'summary_for_diffusion_paper2021.xlsx'
    elif table == 'ensemble_paper':
        expnames_to_table = names_to_table_isbi
        summary_file = config.results / 'model_evaluation' / 'summary_for_ensemble_paper.xlsx'
    else:
        raise RuntimeError(f'{table} not supported')

    # load all relevant data
    mean_results_for_table = dict()
    for sheet in sheets:
        data = pd.read_excel(summary_file, sheet_name=sheet)
        for row in data.iterrows():
            model_name = row[1]['Unnamed: 0']

            this_metric_mean = row[1]['mean ' + metric]
            if model_name not in mean_results_for_table:
                mean_results_for_table[model_name] = []
            mean_results_for_table[model_name].append(this_metric_mean)

    # condense data for all table rows
    rows = []
    for model_name, latex_str in expnames_to_table.items():

        these_results = []
        for res_model_name, metrics in mean_results_for_table.items():
            if res_model_name[:len(model_name)] == model_name:
                these_results.append(np.array(metrics))

        if len(these_results) < 10:
            print(f'not enough results for {model_name}, expected 10, got {len(these_results)}')
            digit_str = 'N/A & N/A & N/A & N/A \\\\'
        else:
            these_results = np.stack(these_results, axis=0)
            means = np.mean(these_results, axis=0)
            stds = np.std(these_results, axis=0)
            digit_str = ''.join(f' ${n:0.3f} \\pm {s:0.3f}$ &' for n, s in zip(means, stds))[:-1] + '\\\\'
        rows.append(latex_str + digit_str)

    for row in rows:
        print(row)


if __name__ == '__main__':
    print_table('ensemble_paper')
    print('------------------------------')
    print_table('diffusion_paper_313')
    print('------------------------------')
    print_table('diffusion_paper_2021')
