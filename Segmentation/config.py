import configparser
import os
from pathlib import Path

config = configparser.ConfigParser()
config.read('config.ini')


def read_section_as_dict(section):
    section_dict = {}
    options = config.options(section)
    for option in options:
        section_dict[option] = config.get(section, option)
    return section_dict


assert 'paths' in config.sections(), 'Need to add a path section in config.ini'
configs = read_section_as_dict('paths')

assert 'datasets' in configs, 'Need to add a datasets entry to paths section in config.ini'
assert 'models' in configs, 'Need to add a models entry to paths section in config.ini'
assert 'results' in configs, 'Need to add a results entry to paths section in config.ini'
assert 'storage' in configs, 'Need to add a storage entry to paths section in config.ini'

datasets = Path(configs['datasets'])
models = Path(configs['models'])
results = Path(configs['results'])
storage = Path(configs['storage'])

assert 'specs' in config.sections(), 'Need to add a specs section in config.ini'
configs = read_section_as_dict('specs')

assert 'base_batch_size' in configs, 'Need to add a base_batch_size entry to specs section in config.ini'
base_batch_size = int(configs['base_batch_size'])

if not os.path.exists(models):
    os.makedirs(models)
if not os.path.exists(results):
    os.makedirs(results)

ignore_value = 255
