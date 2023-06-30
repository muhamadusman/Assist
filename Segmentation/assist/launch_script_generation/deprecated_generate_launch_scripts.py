"""Rewrite with train for folder name"""
from pathlib import Path


def write_launch_file(file_path: Path, function_call_str: str):
    assert not file_path.exists(), 'launch file already exists'
    print(f'{file_path.stem}: {function_call_str}')
    with open(file_path, 'w') as f:
        f.write('#!/bin/bash \n\n')
        f.write('#SBATCH --gpus 1\n')
        f.write('#SBATCH -t 3-00:00:00\n')
        f.write('#SBATCH -A Berzelius-2023-74\n\n')
        f.write('module load Anaconda/2021.05-nsc1\n')
        f.write('conda activate /proj/assist/users/x_manla/assist-env\n\n')
        f.write('if [ -n $SLURM_JOB_ID ];  then\n')
        f.write("    SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')\n")
        f.write('else\n')
        f.write('    SCRIPT_PATH=$(realpath $0)\n')
        f.write('fi\n')
        f.write('SOFTWARE_NAME=$(basename $SCRIPT_PATH)\n')
        f.write('echo "Running $SOFTWARE_NAME"\n\n')

        f.write(f'PYTHONPATH=. python3.9 {function_call_str} \n')


if __name__ == '__main__':
    count = 1

    root_path = Path(__file__).parent.parent.parent

    with_augmentation = True
    include_synthetic_evaluation_experiments = False
    include_increment_experiments = False
    include_new_synthetic_evaluation_experiments = False
    include_newer_synthetic_evaluation_experiments = False
    include_stylegan_experiments = False
    include_synth_exp_20220806 = True

    if include_synthetic_evaluation_experiments:
        base_call_str = "learning/train.py 0 Brats184 'valid' --include-original"
        if with_augmentation:
            base_call_str += ' --geoaug --coloraug'
        extra_arg_strings = ('',
                             '--include-synthetic',
                             '--include-synthetic --augmentation rotation',
                             '--include-synthetic --single-channel',
                             '--include-synthetic --single-channel --augmentation rotation')

        for extra_arg_string in extra_arg_strings:
            write_launch_file(root_path / f'launch{count}.sh', base_call_str + f' {extra_arg_string}')
            count += 1

        base_call_str = "learning/train.py 0 Brats184 'valid'"
        if with_augmentation:
            base_call_str += ' --geoaug --coloraug'
        extra_arg_strings = ('--include-synthetic',
                             '--include-synthetic --augmentation rotation',
                             '--include-synthetic --single-channel',
                             '--include-synthetic --single-channel --augmentation rotation')

        for extra_arg_string in extra_arg_strings:
            write_launch_file(root_path / f'launch{count}.sh', base_call_str + f' {extra_arg_string}')
            count += 1

    if include_increment_experiments:
        base_call_str = 'learning/train_increment_experiments.py'
        for fraction in (0.2, 0.4, 0.6, 0.8, 1.0):
            extra_arg_string = f'{fraction}'
            write_launch_file(root_path / f'launch{count}.sh', base_call_str + f' {extra_arg_string}')
            count += 1

            extra_arg_string += ' --include-synthetic'
            write_launch_file(root_path / f'launch{count}.sh', base_call_str + f' {extra_arg_string}')
            count += 1

    if include_new_synthetic_evaluation_experiments:
        base_call_str = "learning/train.py 0 Brats184 'valid' --include-synthetic --single-channel --full-annotations"
        if with_augmentation:
            base_call_str += ' --geoaug --coloraug'
        extra_arg_strings = ('',
                             '--augmentation mirror',
                             '--include-original',
                             '--include-original --augmentation mirror')

        for extra_arg_string in extra_arg_strings:
            write_launch_file(root_path / f'launch{count}.sh', base_call_str + f' {extra_arg_string}')
            count += 1

        base_call_str = "learning/train.py 0 Brats313 'valid' --include-synthetic --single-channel --full-annotations"
        if with_augmentation:
            base_call_str += ' --geoaug --coloraug'
        extra_arg_strings = ('',
                             '--augmentation mirror',
                             '--include-original',
                             '--include-original --augmentation mirror')

        for extra_arg_string in extra_arg_strings:
            write_launch_file(root_path / f'launch{count}.sh', base_call_str + f' {extra_arg_string}')
            count += 1

    if include_newer_synthetic_evaluation_experiments:
        base_call_str = "learning/train.py 0 Brats313 'valid' --include-synthetic --single-channel --no-pixel-norm" + \
            " --augmentation no"
        extra_arg_strings = ('',
                             '--include-original',
                             ' --geoaug --coloraug',
                             '--include-original  --geoaug --coloraug')

        for extra_arg_string in extra_arg_strings:
            write_launch_file(root_path / f'launch{count}.sh', base_call_str + f' {extra_arg_string}')
            count += 1

        base_call_str = "learning/train.py 0 Brats313 'valid' --include-synthetic --single-channel --augmentation mirror"
        extra_arg_strings = ('',
                             '--include-original',
                             ' --geoaug --coloraug',
                             '--include-original  --geoaug --coloraug')

        for extra_arg_string in extra_arg_strings:
            write_launch_file(root_path / f'launch{count}.sh', base_call_str + f' {extra_arg_string}')
            count += 1

    if include_synth_exp_20220806:
        base_call_str = "learning/train.py 0 Brats313 'valid' --include-synthetic --single-channel --augmentation no"

        extra_arg_strings = ('',
                             '--include-original',
                             ' --geoaug --coloraug',
                             '--include-original  --geoaug --coloraug')

        for special_setting in ('--no-leaky-relu', '--no-smoothing', '--no-repeat'):
            for extra_arg_string in extra_arg_strings:
                write_launch_file(root_path / f'launch{count}.sh', base_call_str +
                                  f' {special_setting} {extra_arg_string}')
                count += 1

    if include_stylegan_experiments:
        count = 1
        base_call_str = "learning/train.py 0 Brats313 'valid' --include-synthetic --single-channel --augmentation no " + \
            "--stylegan"
        extra_arg_strings = ('',
                             '--include-original',
                             ' --geoaug --coloraug',
                             '--include-original  --geoaug --coloraug')

        for extra_arg_string in extra_arg_strings:
            write_launch_file(root_path / f'stylegan{count}.sh', base_call_str + f' {extra_arg_string}')
            count += 1
