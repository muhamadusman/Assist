# master_thesis_2023_henrik_traff


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
cd master_thesis_2023_henrik_traff/swin_and_vit
pip install -v -e .
# '-v' means verbose, or more output
# '-e' means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```
For more information see [mmsegmentation installation README](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation)

### nnU-Net
The nnU-Net framework/library needs to be installed.

```
cd master_thesis_2023_henrik_traff/nnunet_
pip install -e .
```
### Changing ~/.bashrc for nnU-Net
nnU-Net needs to know where you intend to save raw data, preprocessed data and trained models. For this you need to set a few of environment variables.
Add the following to the ~/.bashrc.

```
export nnUNet_raw_data_base="/local/data1/hentr783/thesis_data/nnUNet_raw_data_base"
export nnUNet_preprocessed="/local/data1/hentr783/thesis_data/nnUNet_preprocessed"
export RESULTS_FOLDER="/local/data1/hentr783/thesis_data/nnUNet_trained_models"
```

Also, to change the number of workers employed for data augmentation and validation add
these lines as well.
```
export nnUNet_n_proc_DA=1 # Data Augmentation
export nnUNet_def_n_proc=1 # For validation 
```

### Changing the code for your computer
Throughout most of the scripts used in this thesis, hard coded file path variables
have been used to facilitate implementation. These needs to be changed for other usecases and
other computers. 

In the repository, I have tried to mark these variables with the comment **# Change path for your purposes**

## Prepare Datasets
From the original data pool, **MICCAI_BraTS2020_TrainingData_slices**, create 30 datasets, 6 for each run to be used to train and evaluate the models: e.g., 2dslices_50_001, 2dslices_60_001, ..., 2dslices_90_001, 2dslices_100_001. Need to be in the same directory as the script to run the code. 
```
python create_exp_datasets.py 
```
## Prepare Datasets for MMsegmentation
All 30 datasets created in step 2 need to be prepared for mmsegmentation and the transformer models. This in turn creates 30 new datasets with the same data and split, only structured differently. 
```
python mmseg_data_prep.py 
```
## Prepare Datasets for nnU-Net
All 30 datasets created in step 2 need to be prepared for nnu-net framework. This in turn creates 30 new datasets with the same data and split, only structured differently so that it is compatible with the nnunet framework.
**nnunet_data_prep_and_struct.py**:  prepares a dataset for nnunet, also executes the preprocessing command inherent to the nnunet framework. 
```
python nnunet_data_prep_and_struct.py 
```
After this is done the script **change_batch_size.py** can be called to change batch size to 8.

## Train the models
In the folder **experiment_runs**, some scripts can be found that can be used to train the models, if It does not work than probably some filepath variables need to be changed.

## Generate segmentation masks
Each of the scripts used to generate segmentation masks expect the model file to be named something specific.
For the Transformers and U-Net, its best.pth. For the nnU-Net its best.model, best.model.pkl.
So after a model has been trained, the best model file needs to manually renamed to work with the script that generates segmentation masks.
### Transformer
Run /local/data1/hentr783/master_thesis_2023_henrik_traff/swin_and_vit_/segmentation_prep.py, to get 3D predicted and ground truth segmentation mask. 
These scripts might need to be changed to work for your usecase.
### U-Net
First Run the following script, /local/data1/hentr783/master_thesis_2023_henrik_traff/generate_predictions/unet_predict_exp1.py, to get 2D prediction masks on test, then run , /local/data1/hentr783/master_thesis_2023_henrik_traff/evaluation/segmentation_prep.py , to yield 3D predicted and ground truth segmentation mask.
These scripts might need to be changed to work for your usecase.

### nnU-Net
First Run the following script, /local/data1/hentr783/master_thesis_2023_henrik_traff/generate_predictions/nnunet_predict_exp1_2.py, to get 2D prediction masks on test, then run , /local/data1/hentr783/master_thesis_2023_henrik_traff/evaluation/segmentation_prep.py , to yield 3D predicted and ground truth segmentation mask.
These scripts might need to be changed to work for your usecase.

## Evaluation 
All scripts for generating evaluation metrics can be found in the **evaluation** folder. In each of the scripts, to code will mostly likely need to be changed to work for your usecase.
