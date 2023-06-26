The code is an extension of StyleGAN3 provided by [NVlabs](https://github.com/NVlabs/stylegan3). By changing the provided files in the directories and then running the `train.py` file you can train the model

A docker container is available which can be converted to a singularity container from the given command (singularity build stylegan3_pytorch_tf_latest.sif Dockerfile)

launch.sh file can be executed using sbatch or the command (singularity exec --no-home --nv -H homeDirectory/  -B baseDirectory/  stylegan3_pytorch_tf_latest.sif python train.py --outdir=training_stylegan3-runs-313_gamm-5_fullAnnotations --cfg=stylegan3-t --data=stylegan3_modified/data/brats2020_313subjects_noaugmentation_singlesegmentationchannel_min15percentcoverage/T1/ --gpus=4 --batch=32 --gamma=5 --mirror=1) can be used to initiate the training 
different Gamma values can used to train
