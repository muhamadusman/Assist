
## Getting started

After having cloned this repository to your computer, a few steps need to be performed.


### Conda Environment
The Conda environment with the following packages and libraries 
was used to execute the master thesis.

+ **PyTorch (1.13.1)**
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia 
```
+ **Pytorch ignite**
```
conda install ignite -c pytorch 
```
+ **Seaborn**
```
conda install -c anaconda seaborn  
```
+ **Nibabel**
```
conda install -c conda-forge nibabel   
```
+ **ImageIO**
```
conda install -c conda-forge imageio   
```
+ **monai**
```
conda install -c conda-forge monai  
```
+ **elasticdeform**
```
pip install elasticdeform  
```

### MMSegmentation
The mmsegmentation framework/library needs to be installed.
- Install MMCV using MIM.
```
pip install -U openmim
mim install mmcv-full
```
- Install MMSegmentation
```
cd swin_and_vit
pip install -v -e .
# '-v' means verbose, or more output
# '-e' means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```
For more information see [mmsegmentation installation README](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation)

### Changing the code for your computer
Throughout most of the scripts used in this thesis, hard coded file path variables
have been used to facilitate implementation. These needs to be changed for other usecases and
other computers. 

In the repository, I have tried to mark these variables with the comment **# Change path for your purposes**

## Prepare Datasets
Prepare the dataset 
```
python create_exp_datasets.py 
```
## Prepare Datasets for MMsegmentation
The dataset created in step 2 needs to be prepared for transformer models. This in turn creates new dataset with the same data and split, only structured differently. 
```
python mmseg_data_prep.py 
```
## Train the models
In the folder **experiment_runs**, some scripts can be found that can be used to train the models, if It does not work than probably some filepath variables need to be changed.

## Inference/Evaluation 
In the folder Inference the script can be used to generate the Dice score of test data. The dice score will be based on 3D volume
