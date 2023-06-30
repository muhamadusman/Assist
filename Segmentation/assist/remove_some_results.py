import shutil

import config
from learning.utils import get_checkpoint

res_path = config.results / 'model_evaluation'

for res_folder in res_path.iterdir():
    prediction_folder = res_folder / 'test_predictions'
    if prediction_folder.exists():
        print(prediction_folder)
        shutil.rmtree(prediction_folder)

model_path = config.models
for model_folder in model_path.iterdir():
    model_checkpoint, val_metric, epoch = get_checkpoint(model_folder, model_key='last')
    if model_checkpoint is not None:
        print(model_folder)
        model_checkpoint.unlink()
