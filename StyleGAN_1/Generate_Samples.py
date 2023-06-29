# This file is used to generate Samples from a Trained StyleGAN_1 model


import os
import dnnlib
import dnnlib.tflib as tflib
import numpy as np
import PIL.Image
import pickle

# Generate and save images with different random seeds
num_images = 100000

print ("Generating Samples")
# Path to pre-trained StyleGAN1 .pkl file
pkl_path = 'network-snapshot-016526.pkl'

# Path to output directory
output_dir = 'Samples_StyelGAN1'

# Create output directories if they don't exist
for channel_name in ['T1', 'T1ce', 'flair', 'T2', 'seg']:
    channel_dir = os.path.join(output_dir, channel_name)
    os.makedirs(channel_dir, exist_ok=True)

# Load pre-trained StyleGAN1 .pkl file
tflib.init_tf()
with open(pkl_path, 'rb') as f:
    _G, _D, Gs = pickle.load(f)


used_seeds = set() # set to keep track of used seeds
for i in range(num_images):
    # Generate a random seed that hasn't been used before
    while True:
        seed = np.random.randint(100000000)
        if seed not in used_seeds:
            used_seeds.add(seed)
            break
    
    # Generate random latent vector with the chosen seed
    rnd = np.random.RandomState(seed)
    latents = rnd.randn(1, Gs.input_shape[1])

    # Generate image from latent vector
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

    # Save each channel separately as a grayscale image
    for j, channel_name in enumerate(['T1', 'T1ce', 'flair', 'T2', 'seg']):
        filename = f'image_{i}_{channel_name}.png'
        channel_dir = os.path.join(output_dir, channel_name)
        filepath = os.path.join(channel_dir, filename)
        channel = images[0, :, :, j]
        PIL.Image.fromarray(channel, mode='L').save(filepath)
    
    print ("Image "+i.__str__()+" Generated")

