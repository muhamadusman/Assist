import pandas as pd
import numpy as np

import config


def include_fct_2021(model_folder_name):
    return 'brats21' in model_folder_name.lower() or 'Brats2021Train' in model_folder_name or \
        'brats2021' in model_folder_name.lower()


def include_fct_same(model_folder_name):
    return 'sameexp' in model_folder_name.lower()


def include_fct_stylegans(model_folder_name):
    deprecated_datasets = {'313_out_stylegan3_gamma2_025000kimg',
                           '313_out_stylegan3_gamma2_018200kimg',
                           '313_out_stylegan3_gamma5_025000kimg'}
    for dep in deprecated_datasets:
        if dep in model_folder_name.lower():
            return False
    return 'stylegan' in model_folder_name.lower()


def include_fct_extra_ensembles(model_folder_name):
    patterns = ('1GAN', '5GAN', '10GAN')
    for pattern in patterns:
        if pattern in model_folder_name:
            return True
    return False


def include_fct_isbi(model_folder_name):
    for root_model_name in ('UNet-202209-Brats313-256-geoaug-imaug-valid-lr5.0e-02',
                            'UNet-202209-Brats313-synthetic_1GAN-256-geoaug-imaug-valid-lr5.0e-02',
                            'UNet-202209-Brats313-synthetic_5GANs-256-geoaug-imaug-valid-lr5.0e-02',
                            'UNet-202209-Brats313-synthetic_10GANs-256-geoaug-imaug-valid-lr5.0e-02',
                            'UNet-202209-Brats313-synthetic_20GANs-256-geoaug-imaug-valid-lr5.0e-02',
                            'UNet-202209-synthetic_1GAN-256-geoaug-imaug-valid-lr5.0e-02',
                            'UNet-202209-synthetic_5GANs-256-geoaug-imaug-valid-lr5.0e-02',
                            'UNet-202209-synthetic_10GANs-256-geoaug-imaug-valid-lr5.0e-02',
                            'UNet-202209-synthetic_20GANs-256-geoaug-imaug-valid-lr5.0e-02'):
        if model_folder_name[:len(root_model_name)] == root_model_name:
            return True
    return False


def include_fct_diffusion(model_folder_name):
    return 'diffusion' in model_folder_name


def include_fct_pgans(model_folder_name):
    if 'BratsDiff' in model_folder_name or 'SameExp' in model_folder_name:
        return False
    return not (include_fct_stylegans(model_folder_name) or
                include_fct_extra_ensembles(model_folder_name) or
                include_fct_isbi(model_folder_name) or
                include_fct_diffusion(model_folder_name))


def include_fct_diffusion_paper313(model_folder_name):
    from assist.experiment_names_lookup import diffusion_paper_313_datasets
    diffusion_paper_313_datasets['UNet-202209-Brats313-256'] = 'None'
    for root_name in diffusion_paper_313_datasets:
        if root_name in model_folder_name:
            return True
    return False


def include_fct_diffusion_paper2021(model_folder_name):
    from assist.experiment_names_lookup import diffusion_paper_2021_datasets
    diffusion_paper_2021_datasets['UNet-202209-Brats2021Train-256'] = 'None'
    for root_name in diffusion_paper_2021_datasets:
        if root_name in model_folder_name:
            return True
    return False


def load_result(result_file, sheets):
    mean_results = dict()
    std_results = dict()

    all_data = [pd.read_excel(result_file, sheet_name=sheet) for sheet in sheets]
    for i, sheet in enumerate(sheets):
        mean_results[sheet] = dict()
        std_results[sheet] = dict()
        data = all_data[i]

        for column in data.columns:
            if column != 'Unnamed: 0':
                mean_results[sheet][column] = np.nanmean(data[column])
                std_results[sheet][column] = np.nanstd(data[column])

    mean_results['mean'] = dict()
    for sheet in sheets:
        if sheet == 'foreground':
            continue

        for metric, val in mean_results[sheet].items():
            if metric not in mean_results['mean']:
                mean_results['mean'][metric] = 0
            mean_results['mean'][metric] += val / (len(sheets) - 1)

    std_results['mean'] = dict()
    for metric in mean_results['mean']:
        this_metric = np.array(all_data[1][metric]).astype(np.float64)
        for i in range(2, len(all_data)):
            this_metric += np.array(all_data[i][metric]).astype(np.float64)

        this_metric /= (len(all_data) - 1)
        std_results['mean'][metric] = np.nanstd(this_metric)
    return mean_results, std_results


def run():
    # save_file_name = 'summary_diffusion.xlsx'
    # include_fct = include_fct_diffusion

    # save_file_name = 'summary_pgans.xlsx'
    # include_fct = include_fct_pgans

    # save_file_name = 'summary_stylegan.xlsx'
    # include_fct = include_fct_stylegans

    # save_file_name = 'summary_ensembles.xlsx'
    # include_fct = include_fct_extra_ensembles

    # save_file_name = 'summary_same.xlsx'
    # include_fct = include_fct_same

    # save_file_name = 'summary_for_ensemble_paper.xlsx'
    # include_fct = include_fct_isbi

    save_file_name = 'summary_for_diffusion_paper313.xlsx'
    include_fct = include_fct_diffusion_paper313

    # save_file_name = 'summary_for_diffusion_paper2021.xlsx'
    # include_fct = include_fct_diffusion_paper2021

    sheets = ('foreground', 'necrotic_non_enhancing_tumor_co', 'peritumoral_edema', 'gd_enhancing_tumor')

    save_folder = config.results / 'model_evaluation'

    all_results = dict()
    for model_result_folder in save_folder.iterdir():
        test_result_file = model_result_folder / 'test_result.xlsx'
        if not include_fct(model_result_folder.name):
            print(f'{model_result_folder.name} not to be included according to include_fct, skipping')
            continue
        if test_result_file.exists():
            all_results[model_result_folder.name] = load_result(test_result_file, sheets)

    with pd.ExcelWriter(save_folder / save_file_name, engine='xlsxwriter') as writer:
        for sheet in sheets + ('mean', ):
            lines = []
            for name, (mean_res, std_res) in all_results.items():
                line = pd.DataFrame(index=[name])
                for metric, val in mean_res[sheet].items():
                    line['mean ' + metric] = val
                for metric, val in std_res[sheet].items():
                    line['std ' + metric] = val

                lines.append(line)

            pd_data = pd.concat(lines).sort_index()

            pd_data.to_excel(writer, sheet_name=sheet)


if __name__ == '__main__':
    run()
