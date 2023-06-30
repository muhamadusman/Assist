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


def write_launch_all_file(file_path: Path, count: int):
    with open(file_path, 'w') as f:
        for i in range(1, count + 1):
            f.write(f'sbatch stylegan{i}.sh \n')


if __name__ == '__main__':
    count = 1

    root_path = Path(__file__).parent.parent.parent

    stylegan_folders_184 = ['184_styleGAN2_Gamma2_013000kimg',
                            '184_styleGAN2_Gamma5_020000kimg',
                            '184_styleGAN2_Gamma8_025000kimg',
                            '184_styleGAN3_Gamma2_004400kimg',
                            '184_styleGAN3_Gamma5_011200kimg',
                            '184_styleGAN3_Gamma8_025000kimg',
                            ]

    # FakeImages_styleGAN3_5Channel already trained with old scripts
    stylegan_folders_313 = ['313_out_stylegan2_gamma2_025000kimg',
                            '313_out_stylegan2_gamma5_025000kimg',
                            '313_out_stylegan2_gamma8_025000kimg',
                            'out_stylegan3_gamma2_025000kimg_noaug',
                            'out_stylegan3_gamma5_025000kimg_noaug',
                            'out_stylegan3_gamma8_025000kimg_noaug',
                            'styleGAN2_fullAnonation_Gamma2_012600kimg',
                            'styleGAN2_fullAnonation_Gamma5_025000kimg',
                            'styleGAN2_fullAnonation_Gamma8_025000kimg',
                            'styleGAN3_fullAnonation_Gamma2_025000kimg',
                            'styleGAN3_fullAnonation_Gamma5_025000kimg',
                            'styleGAN3_fullAnonation_Gamma8_025000kimg']

    extra_args_for_each_folder = ('',
                                  '--include-original',
                                  ' --geoaug --coloraug',
                                  '--include-original  --geoaug --coloraug')

    count = 1
    for synthetic_folder_name in stylegan_folders_184:
        base_call_str = f"learning/train_from_folder_name.py 0 Brats184 'valid' {synthetic_folder_name}"

        for extra_arg_string in extra_args_for_each_folder:
            write_launch_file(root_path / f'stylegan{count}.sh', base_call_str + f' {extra_arg_string}')
            count += 1

    for synthetic_folder_name in stylegan_folders_313:
        base_call_str = f"learning/train_from_folder_name.py 0 Brats313 'valid' {synthetic_folder_name}"

        for extra_arg_string in extra_args_for_each_folder:
            write_launch_file(root_path / f'stylegan{count}.sh', base_call_str + f' {extra_arg_string}')
            count += 1

    write_launch_all_file(root_path / 'launch_all_stylegans.sh', count - 1)
