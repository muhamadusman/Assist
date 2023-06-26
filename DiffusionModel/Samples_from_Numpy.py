import numpy as np
from PIL import Image

file = "samples_16x256x256x5.npy"
images = np.load(file)

final_images = []
for i in range(5):
    final_image = np.concatenate([Image.fromarray(images[j][:, :, i]).convert("L") for j in range(1, 9)], axis=0)
    final_images.append(final_image)

final_image = np.concatenate(final_images, axis=1)

img = Image.fromarray(final_image)
img.save("Sample_model_1200000.png")
