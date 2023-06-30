from assist.experiment_names_lookup import trained_models
import config
from shutil import copyfile

models_to_move = trained_models['ensembles_isbi']

res_path = config.results / 'model_evaluation'
moved_path = config.results / 'tmp_results'

moved_path.mkdir(exist_ok=True)

for model_name in models_to_move:
    if 'fullrandomsplit' in model_name or ('Brats184' not in model_name and 'Brats313' not in model_name):
        model_choice = 'val_mean_dice'
    else:
        model_choice = 'val_macro_dice_mean'

    copyfile(res_path / f'{model_name}-{model_choice}' / 'test_result.xlsx',
             moved_path / f'{model_name[12:].replace("-256-geoaug-imaug-valid-lr5.0e-02", "")}.xlsx')
