from typing import Optional

import config
import pandas as pd

explicit_folder_coding = {
    'StyelaGAN2_Gamma2': 'StyleGAN 2 $\\gamma$: 2',
    'StyelaGAN2_Gamma5': 'StyleGAN 2 $\\gamma$: 5',
    'StyelaGAN2_Gamma8': 'StyleGAN 2 $\\gamma$: 8',
    'DiffusionModel_Brats21': 'Diffusion',
    'BRATS2021_synthetic_1195subjects_noaugmentation_singlesegmentationchannel': 'pGAN single'
}


def decode_model_name(model_name: str) -> Optional[str]:
    success = False

    if 'Brats2021Train' in model_name:
        latex_str = '\\checkmark &'
        success = True
    else:
        latex_str = ' &'

    if '-geoaug-imaug' in model_name:
        latex_str += ' \\checkmark &'
    else:
        latex_str += ' &'

    hardcoded = False
    for dset_name, dset_gan_str in explicit_folder_coding.items():
        if dset_name in model_name:
            latex_str += ' ' + dset_gan_str + ' &'
            hardcoded = True
            success = True
            break

    if not hardcoded:
        if 'Brats2021Train-256' in model_name:
            latex_str += ' &'
        else:
            success = False

    if success:
        return latex_str
    else:
        print(f'unable to decode {model_name}')
        return None


sheets = ('gd_enhancing_tumor', 'peritumoral_edema', 'necrotic_non_enhancing_tumor_co', 'mean')
metric = 'dice'
add_stds = False

debug = False
summary_file = config.results / 'model_evaluation' / 'summary_Brats2021.xlsx'

mean_results_for_table = dict()
std_results_for_table = dict()
for sheet in sheets:
    data = pd.read_excel(summary_file, sheet_name=sheet)
    for row in data.iterrows():
        model_name = row[1]['Unnamed: 0']

        this_metric_mean = row[1]['mean ' + metric]
        if model_name not in mean_results_for_table:
            mean_results_for_table[model_name] = []
        mean_results_for_table[model_name].append(this_metric_mean)

        if add_stds:
            this_metric_std = row[1]['std ' + metric]
            if model_name not in std_results_for_table:
                std_results_for_table[model_name] = []
            std_results_for_table[model_name].append(this_metric_std)


rows = []
for model_name, numbers in mean_results_for_table.items():
    latex_str = decode_model_name(model_name)
    if latex_str is not None:
        if add_stds:
            stds = std_results_for_table[model_name]
            digit_str = ''.join(f' ${n:0.3f} \\pm {s:0.3f}$ &' for n, s in zip(numbers, stds))[:-1] + '\\\\'
        else:
            digit_str = ''.join(f' {n:0.3f} &' for n in numbers)[:-1] + '\\\\'
        rows.append(latex_str + digit_str)

for row in sorted(rows):
    print(row)
