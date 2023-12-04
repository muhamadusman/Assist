# Assist 2d segmentation code


## Install

It is recommended to run the code inside a python virtualenv.

* Install system dependencies

```
sudo apt-get install htop virtualenv tmux python3-pip python3-tk  python3.9 python3.9-dev python3.9-distutils
```


* Install virtualenvwrapper

```
pip3 install virtualenv virtualenvwrapper
mkdir ~/.virtualenvs
echo 'export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3' >> ~/.bashrc
echo 'export WORKON_HOME=$HOME/.virtualenvs' >> ~/.bashrc
echo 'export VIRTUALENVWRAPPER_VIRTUALENV=$HOME/.local/bin/virtualenv' >> ~/.bashrc
echo 'source $HOME/.local/bin/virtualenvwrapper.sh' >> ~/.bashrc
source ~/.bashrc
source ~/.local/bin/virtualenvwrapper.sh
```

* Setup virtualenv

```
mkvirtualenv --python=python3.9 assist
```

* Install all python dependencies

```bash
pip install -r requirements.txt
```

* Create a config.ini file, with the following content (modify to your paths)

```
[paths]
storage: path_to_proj/datasets
datasets: /scratch/local/datasets # do not change this for berzelius
models: path_to_proj/models
results: path_to_proj/results

[specs]
base_batch_size: 48
```

## Conda
If you prefer to use conda for environment there is an environment file included



# Workflow
- Install environment on berzelius (recommended for training) and local computer (recommended for running models)
- Place synthetic dataset compressed to .tar files in storage path defined in config.ini (all synthetic datasets up until now can be found compressed in /proj/assist/users/x_manla/datasets)
- assist/launch_script_generations contains script to generate launch scripts, eg. assist/launch_script_generations/generate_launch_scripts_diffusion_paper.py generates all launch scripts for the experiments in "A Performance Analysis of Diffusion Model and GANs for Synthetic Brain Tumor MRI Data Generation". Running this script on berzelius and then ```source diffusion.sh``` will schedule training for all experiments. If you need to add new experiments or redo some old trainings, modify the script
- During training, you can checkt slurm files for progress. There is also a helper script, assist/check_trainings.py that summarizes training progress. This is helpful to see if any trainings have crashed (very uncommon). Trainings are done after 72 hours.
- [Optional] download models to local computer
- Run models on test set, learning/run.py will run all models were there isn't any results file yet. This script can also be modified to only run certain models only.
- Summarize results: /analysis/create_summary.py will create a .xlsx file with result metrics for all models included. The included models are chosen by defining and "include function", look at the code in create_summary.py to see how this works. For experiments in "A Performance Analysis of Diffusion Model and GANs for Synthetic Brain Tumor MRI Data Generation", the include function look at the variables diffusion_paper_313_datasets and diffusion_paper_2021_datasets defined in /assist/experiment_names_lookup.py. Modify these if you need to add new experiments.
- Run /analysis/create_table_for_papers.py to summarize results in a folder, these can be copy-pasted into latex.
- To create visualizations,run learning/run.py again for the models where you want to show results. Change the option skip_image_creation to False. The script /assist/gather_plots_for_overleaf.py can be used to copy specific files to facilitate uploading to overleaf.
