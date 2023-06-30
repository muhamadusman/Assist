
import shutil
from pathlib import Path

import config
from data.dataset import BratsTest

im_name = 'Subject_009_slice_76_T1.png'
res_folder = Path('/home/mans/Downloads/to_overleaf')
res_folder.mkdir(exist_ok=True)

dataset = BratsTest(None, None, None)
expnames = ['UNet-202209-Brats313-256-geoaug-imaug-valid-lr5.0e-02-0-val_macro_dice_mean',
            'UNet-202209-Brats313-synthetic_1GAN-256-geoaug-imaug-valid-lr5.0e-02-6-val_macro_dice_mean',
            'UNet-202209-Brats313-synthetic_5GANs-256-geoaug-imaug-valid-lr5.0e-02-6-val_macro_dice_mean',
            'UNet-202209-Brats313-synthetic_10GANs-256-geoaug-imaug-valid-lr5.0e-02-9-val_macro_dice_mean',
            'UNet-202209-Brats313-synthetic_20GANs-256-geoaug-imaug-valid-lr5.0e-02-0-val_macro_dice_mean',
            'UNet-202209-synthetic_1GAN-256-geoaug-imaug-valid-lr5.0e-02-4-val_mean_dice',
            'UNet-202209-synthetic_5GANs-256-geoaug-imaug-valid-lr5.0e-02-3-val_mean_dice',
            'UNet-202209-synthetic_10GANs-256-geoaug-imaug-valid-lr5.0e-02-6-val_mean_dice',
            'UNet-202209-synthetic_20GANs-256-geoaug-imaug-valid-lr5.0e-02-0-val_mean_dice']

shutil.copyfile(dataset.root.parent / 'plots' / im_name, res_folder / f'gt_{im_name}')

for expname in expnames:
    save_folder = config.results / 'model_evaluation' / expname
    exp_identifier = '-'.join(expname.split('-')[2:-8])
    shutil.copyfile(save_folder / 'test_images' / im_name, res_folder / f'{exp_identifier}_{im_name}')
