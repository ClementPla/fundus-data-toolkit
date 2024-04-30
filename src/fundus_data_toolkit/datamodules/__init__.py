import os
from collections import namedtuple
from pathlib import Path

import yaml

CLASSIF_PATHS = None
SEG_PATHS = None
yaml_file = None

cpath = Path(os.path.dirname(os.path.realpath(__file__)))
try:
    yaml_path = Path('config/config.yaml')
    with open(yaml_path) as f:
        yaml_file = yaml.load(f, Loader=yaml.FullLoader)
except FileNotFoundError:
    yaml_path = cpath /'../../..'/ Path('config/config.yaml')
    with open(yaml_path) as f:
        yaml_file = yaml.load(f, Loader=yaml.FullLoader)
    


if yaml_file:
    if 'Classification' in yaml_file:
        CLASSIF_PATHS = namedtuple('ClassificationPaths', yaml_file['Classification'].keys())(**yaml_file['Classification'])
    if 'Segmentation' in yaml_file:
        SEG_PATHS = namedtuple('SegmentationPaths', yaml_file['Segmentation'].keys())(**yaml_file['Segmentation'])