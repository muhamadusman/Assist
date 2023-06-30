import numpy as np
from data.dataset import Brats184, Brats313

brats184 = Brats184(None, None, None)
brats313 = Brats313(None, None, None)

subjects184 = brats184.get_available_subjects()
subjects313 = brats313.get_available_subjects()

if len(set(subjects184).intersection(set(subjects313))) == len(subjects184):
    print('Brats184 is a subset of Brats313')


test_subjects = np.random.choice(subjects184, 10)

for subject in test_subjects:
    indices184 = brats184.get_indices_for_subject(subject)
    indices313 = brats313.get_indices_for_subject(subject)

    assert len(indices184) == len(indices313)

    for i184, i313 in zip(indices184, indices313):
        im184, target184 = brats184[i184]
        im313, target313 = brats313[i313]

        assert np.all(im184 == im313)
        assert np.all(target184 == target313)

print('Success')
