#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=muhammad.usman@liu.se
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -t 3-00:00:00

module load Anaconda/2021.05-nsc1
conda activate /proj/assist/users/x_muhak/.conda/envs/pytorch_env

echo "Training Started"

python /proj/assist/users/x_muhak/assist-vision/henrik/swin_and_vit_/1_ScientificData/Inference_files/InferenceFiles_Augmentation/Brats21.py

echo "Program Terminated"
