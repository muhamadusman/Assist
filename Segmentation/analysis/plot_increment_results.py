
from typing import Optional, Tuple

import config
import pandas as pd
from matplotlib import pyplot as plt


def decode_model_name(model_name: str) -> Tuple[bool, Optional[bool], Optional[int]]:
    if 'BratsDiff' not in model_name:
        return False, None, None

    n_subjects = int(model_name.replace('_', '-').split('-')[3])
    with_synthetic = 'Gan' in model_name
    return True, with_synthetic, n_subjects


save_figs = True
show_figs = False
sheets = ('gd_enhancing_tumor', 'peritumoral_edema', 'necrotic_non_enhancing_tumor_co', 'mean')
plot_titles = ('Gadolinium enhancing tumor', 'Peritumoral edema', 'Necrotic and non-enhancing tumor core', 'Mean')
full_training_vals = (0.798, 0.788, 0.603, 0.730)

metric = 'dice'

summary_file = config.results / 'model_evaluation' / 'summary.xlsx'
if save_figs:
    save_folder = config.results / 'increment_results'
    save_folder.mkdir(exist_ok=True)

results_for_table = dict()
for sheet in sheets:
    data = pd.read_excel(summary_file, sheet_name=sheet)
    for row in data.iterrows():
        model_name = row[1]['Unnamed: 0']
        this_metric = row[1][metric]

        if model_name not in results_for_table:
            results_for_table[model_name] = []
        results_for_table[model_name].append(this_metric)

for label_index, label_name in enumerate(sheets):
    synthetic = dict()
    no_synthetic = dict()
    for model_name, numbers in results_for_table.items():
        success, with_synthetic, n_subjects = decode_model_name(model_name)
        if not success:
            continue
        this_dice = results_for_table[model_name][label_index]
        if with_synthetic:
            synthetic[n_subjects] = this_dice
        else:
            no_synthetic[n_subjects] = this_dice

    synthetic_x = sorted(synthetic.keys())
    synthetic_y = [synthetic[key] for key in synthetic_x]

    no_synthetic_x = sorted(no_synthetic.keys())
    no_synthetic_y = [no_synthetic[key] for key in no_synthetic_x]

    plt.plot(synthetic_x, synthetic_y, '-*', label='with synthetic')
    plt.plot(no_synthetic_x, no_synthetic_y, '-*', label='without synthetic')
    # plt.plot([min(synthetic_x)-1, max(synthetic_x) + 1],
    #         [full_training_vals[label_index], full_training_vals[label_index]], '--', color='k')
    plt.xlabel('number of subjects')
    plt.ylabel('Dice')
    plt.xlim([min(synthetic_x)-1, max(synthetic_x) + 1])
    plt.title(plot_titles[label_index])
    plt.legend()

    if save_figs:
        plt.savefig(save_folder / f'increment_{label_name}.png', bbox_inches='tight')

    if show_figs:
        plt.show()
    else:
        plt.close()
