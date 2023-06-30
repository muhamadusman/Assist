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
            f.write(f'sbatch launch_all_diffusion_paper{i}.sh \n')


if __name__ == '__main__':
    start_seed = 0  # set to non-zero if models have already been trained
    repeats = 10  # seed in range(start_seed, start_seed+repeats)

    root_path = Path(__file__).parent.parent.parent

    synthetic_folders_313 = \
        ['None',  # no synthetic
         'synthetic_1GAN',
         '313_out_stylegan2_gamma8_025000kimg',
         'out_stylegan3_gamma8_025000kimg_noaug',
         'Sample_StyleGAN1',
         'diffusion_313_dataset1']

    synthetic_folders_2021 = ['None',
                              'BRATS2021_synthetic_1195subjects_noaugmentation_singlesegmentationchannel',
                              'BRATS21_StyelaGAN2_Gamma8',
                              'BRATS21_StyelaGAN3_Gamma10',
                              'DiffusionModel_Brats21',
                              'Generated_Images_StyleGAN1_BRATS21']

    extra_args_for_each_folder = ('',
                                  '--include-original',
                                  ' --geoaug --coloraug',
                                  '--include-original  --geoaug --coloraug')

    count = 1
    for synthetic_folder_name in synthetic_folders_313:
        for seed in range(start_seed, repeats):
            base_call_str = f"learning/train_from_folder_name.py {seed} Brats313 'valid' {synthetic_folder_name}"
            for extra_arg_string in extra_args_for_each_folder:
                if synthetic_folder_name == 'None' and 'include-original' not in extra_arg_string:
                    continue
                write_launch_file(root_path / f'diffusion_paper{count}.sh', base_call_str + f' {extra_arg_string}')
                count += 1

    for synthetic_folder_name in synthetic_folders_2021:
        for seed in range(start_seed, start_seed + repeats):
            base_call_str = f"learning/train_from_folder_name.py {seed} Brats2021 'valid' {synthetic_folder_name}"
            for extra_arg_string in extra_args_for_each_folder:
                if synthetic_folder_name is None and 'include-original' not in extra_arg_string:
                    continue
                write_launch_file(root_path / f'diffusion_paper{count}.sh', base_call_str + f' {extra_arg_string}')
                count += 1

    write_launch_all_file(root_path / 'diffusion_paper.sh', count - 1)
