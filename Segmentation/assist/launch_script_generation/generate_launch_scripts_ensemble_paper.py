from pathlib import Path


def write_launch_file(file_path: Path, function_call_str: str, on_berzelius: bool):
    assert not file_path.exists(), 'launch file already exists'
    print(f'{file_path.stem}: {function_call_str}')
    with open(file_path, 'w') as f:
        f.write('#!/bin/bash \n\n')
        if on_berzelius:
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


def write_launch_all_file(file_path: Path, count: int):
    with open(file_path, 'w') as f:
        for i in range(1, count + 1):
            f.write(f'sbatch ensemble{i}.sh \n')


if __name__ == '__main__':
    on_berzelius = True

    repeats = 10

    root_path = Path(__file__).parent.parent.parent

    ensemble_folders = ['None',  # no synthetic
                        'synthetic_1GAN',
                        'synthetic_5GANs',
                        'synthetic_10GANs',
                        'synthetic_20GANs']

    extra_args_for_each_folder = ('',
                                  '--include-original',
                                  ' --geoaug --coloraug',
                                  '--include-original  --geoaug --coloraug')

    count = 1
    for synthetic_folder_name in ensemble_folders:

        for seed in range(repeats):
            base_call_str = f"learning/train_from_folder_name.py {seed} Brats313 'valid' {synthetic_folder_name}"

            for extra_arg_string in extra_args_for_each_folder:
                if synthetic_folder_name == 'None' and 'include-original' not in extra_arg_string:
                    continue
                write_launch_file(root_path / f'ensemble{count}.sh',
                                  base_call_str + f' {extra_arg_string}', on_berzelius)
                count += 1

    write_launch_all_file(root_path / 'launch_all_ensembles.sh', count - 1)
