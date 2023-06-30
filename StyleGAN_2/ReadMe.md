The code is an extension of [NVlabs](https://github.com/NVlabs/stylegan3) where support for StyleGAN2 is also given. By changing the provided files in the directories and then running the `train.py` file you can train the model

A docker container is available which can be converted to a singularity container from the given command (singularity build stylegan2_pytorch_tf_latest.sif Dockerfile)


Initiate Training : 
launch.sh file can be executed using sbatch or the command (singularity exec --no-home --nv -H homeDirectory/  -B baseDirectory/  stylegan2_pytorch_tf_latest.sif python train.py --outdir=outputDirectory --cfg=stylegan2-t --data=data/T1/ --gpus=4 --batch=32 --gamma=8 --mirror=1) can be used to initiate the training 
different Gamma values can be used to train

Initiate Sampling: 
singularity exec --no-home --nv -H $PWD:homeDirectory/  -B baseDirectory/ stylegan3_pytorch_tf_latest.sif python gen_images.py --outdir=outputDirectory --trunc=1 --seeds=2-100002 --network=savedmodel/network-snapshot-025000.pkl

number of samples can be controlled by changing the value of the seed

