import os
import config


def seed_from_name(name):
    return int(name.name.split('_')[0][4:])


root = config.datasets / 'BRATS_stylegan_313subjects_noaugmentation_singlesegmentationchannel' / 'out'

all_images = dict()
all_seeds = dict()
for image_folder in ('T1', 'T1CE', 'T2', 'FLAIR', 'SEG'):
    modality_folder = root / image_folder
    all_images[image_folder] = sorted(modality_folder.glob('*.png'))
    all_seeds[image_folder] = set(seed_from_name(name) for name in all_images[image_folder])

complete_seeds = set.intersection(*list(all_seeds.values()))

for image_folder in ('T1', 'T1CE', 'T2', 'FLAIR', 'SEG'):
    modality_folder = root / image_folder
    images = sorted(modality_folder.glob('*.png'))
    for image in images:
        if seed_from_name(image) not in complete_seeds:
            os.remove(image)
