import config

model_save_folder = config.models


model_save_folder = config.models
for path_object in model_save_folder.rglob('*'):
    if path_object.is_file() and path_object.suffix == '.ckpt':
        experiment_name = path_object.parts[-5]
        if 'fullrandomsplit' in experiment_name or \
                ('Brats184' not in experiment_name and
                 'Brats313' not in experiment_name and
                 'Brats2021Train' not in experiment_name):
            model_choice = 'val_mean_dice'
        else:
            model_choice = 'val_macro_dice_mean'

        if model_choice not in path_object.name:
            path_object.unlink()

model_save_folder = config.results / 'model_evaluation'
for path_object in model_save_folder.rglob('*'):
    if path_object.is_file() and path_object.suffix == '.npy':
        path_object.unlink()
