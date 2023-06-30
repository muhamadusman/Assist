from typing import Optional

import config
import pandas as pd

explicit_folder_coding = {
    '184_styleGAN2_Gamma2_013000kimg': 'StyleGAN 2 & $\\gamma$: 2',
    '184_styleGAN2_Gamma5_020000kimg': 'StyleGAN 2 & $\\gamma$: 5',
    '184_styleGAN2_Gamma8_025000kimg': 'StyleGAN 2 & $\\gamma$: 8',
    '184_styleGAN3_Gamma2_004400kimg': 'StyleGAN 3 & $\\gamma$: 2',
    '184_styleGAN3_Gamma5_011200kimg': 'StyleGAN 3 & $\\gamma$: 5',
    '184_styleGAN3_Gamma8_025000kimg': 'StyleGAN 3 & $\\gamma$: 8',
    '313_out_stylegan2_gamma2_025000kimg': 'StyleGAN 2 & $\\gamma$: 2',
    '313_out_stylegan2_gamma5_025000kimg': 'StyleGAN 2 & $\\gamma$: 5',
    '313_out_stylegan2_gamma8_025000kimg': 'StyleGAN 2 & $\\gamma$: 8',
    'out_stylegan3_gamma2_025000kimg_noaug': 'StyleGAN 3 & $\\gamma$: 2',
    'out_stylegan3_gamma5_025000kimg_noaug': 'StyleGAN 3 & $\\gamma$: 5',
    'out_stylegan3_gamma8_025000kimg_noaug': 'StyleGAN 3 & $\\gamma$: 8',
    'styleGAN2_fullAnonation_Gamma2_012600kimg': 'StyleGAN 2 & extra, $\\gamma$: 2',
    'styleGAN2_fullAnonation_Gamma5_025000kimg': 'StyleGAN 2 & extra, $\\gamma$: 5',
    'styleGAN2_fullAnonation_Gamma8_025000kimg': 'StyleGAN 2 & extra, $\\gamma$: 8',
    'styleGAN3_fullAnonation_Gamma2_025000kimg': 'StyleGAN 3 & extra, $\\gamma$: 2',
    'styleGAN3_fullAnonation_Gamma5_025000kimg': 'StyleGAN 3 & extra, $\\gamma$: 5',
    'styleGAN3_fullAnonation_Gamma8_025000kimg': 'StyleGAN 3 & extra, $\\gamma$: 8',
    'synthetic_10GANs': 'Ensemble & 10 GANs',
    'synthetic_20GANs': 'Ensemble & 20 GANs',
    'synthetic_1GAN': 'Ensemble & 1 GAN',
    'synthetic_5GANs': 'Ensemble & 5 GAN',
    'BRATS_synthetic_313subjects_noaugmentation_singlesegmentationchannel_sitecondition': 'pGAN & single conditional',
    'diffusion_313_dataset1': 'Diffusion & ',
    'Sample_StyleGAN1': 'StyleGAN 1 & ',
}


def decode_model_name(model_name: str, print_extra_info: bool) -> Optional[str]:
    success = False

    if print_extra_info:
        extra_info = '184' if '184' in model_name else '313'
        if '-geoaug-imaug' in model_name:
            extra_info += ' aug: '
        else:
            extra_info += ' noaug: '
    else:
        extra_info = ''

    if 'Brats184' in model_name or 'Brats313' in model_name:
        latex_str = f'{extra_info} \\checkmark &'
        success = True
    else:
        latex_str = f'{extra_info} &'

    hardcoded = False
    for dset_name, dset_gan_str in explicit_folder_coding.items():
        if dset_name in model_name:
            latex_str += ' ' + dset_gan_str + ' &'
            hardcoded = True
            success = True
            break

    if not hardcoded:
        if 'Gan' in model_name:
            if 'single' in model_name:
                gan_str = ' pGAN & single'
            else:
                gan_str = ' pGAN & multiple'

            if 'nopix' in model_name:
                gan_str += ' nonorm'
            if 'fullannotation' in model_name:
                gan_str += ' extra'
            if 'norepeat' in model_name:
                gan_str += ' no repeat'
            if 'noleakyrelu' in model_name:
                gan_str += ' no leaky relu'
            if 'nosmoothing' in model_name:
                gan_str += ' no smoothing'
            if 'rotation' in model_name:
                gan_str = gan_str + ' rot &'
            elif 'mirror' in model_name:
                gan_str = gan_str + ' mirror &'
            else:
                gan_str = gan_str + ' &'
            latex_str = latex_str + gan_str
            success = True
        else:
            latex_str = latex_str + ' & &'

    if success:
        return latex_str
    else:
        print(f'unable to decode {model_name}')
        return None


sheets = ('gd_enhancing_tumor', 'peritumoral_edema', 'necrotic_non_enhancing_tumor_co', 'mean')
metric = 'dice'
add_stds = False

debug = False
summary_file = config.results / 'model_evaluation' / 'summary_stylegan.xlsx'

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


for start_str in ('184 aug', '313 aug', '184 noaug', '313 noaug'):
    rows = []
    print(f'----------------------  {start_str}  ----------------------')
    for model_name, numbers in mean_results_for_table.items():
        latex_str = decode_model_name(model_name, print_extra_info=True)
        if latex_str is not None and latex_str.startswith(start_str):
            if add_stds:
                stds = std_results_for_table[model_name]
                digit_str = ''.join(f' ${n:0.3f} \\pm {s:0.3f}$ &' for n, s in zip(numbers, stds))[:-1] + '\\\\'
            else:
                digit_str = ''.join(f' {n:0.3f} &' for n in numbers)[:-1] + '\\\\'
            if debug:
                rows.append(latex_str[len(start_str) + 1:] + model_name)
            else:
                rows.append(latex_str[len(start_str) + 1:] + digit_str)

    for row in sorted(rows):
        print(row)
