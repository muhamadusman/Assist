Based on using the progressive growing GAN by Nvidida, https://github.com/tkarras/progressive_growing_of_gans 

Use .yml file to make conda environment 

Modify config.py in the PGGAN code to use your new dataset.

Run train.py in inetiate training. 

Warning: This code does not work with too new graphics cards, like RTX 3090 from the Ampere architecture, as the code is using TF 1.14.
