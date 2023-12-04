The code is based on the repository provided by  [OpenAI](https://github.com/openai/guided-diffusion) which is the codebase for [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)


Inetiate Training :

MODEL_FLAGS="--image_size 256 --num_channels 64 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 32"

python /path_to_proj/scripts/image_train.py --data_dir /path_to_proj/datasets/ $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS


Initiate Sampling :

python /path_to_proj/scripts/image_sample_modified_single.py --model_path /path_to_proj/guided_diffusion/model030000.pt $MODEL_FLAGS $DIFFUSION_FLAGS
