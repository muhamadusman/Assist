The code is based on the repository provided by  [OpenAI](https://github.com/openai/guided-diffusion) which is codebase for [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)


Inetiate Training :

MODEL_FLAGS="--image_size 256 --num_channels 64 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 32"

python scripts/image_train.py --data_dir datasets/313_3Channel_Images_train/ $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS


Inetiate Sampling :

python scripts/image_sample_modified_single.py --model_path guided_diffusion/model030000.pt $MODEL_FLAGS $DIFFUSION_FLAGS
