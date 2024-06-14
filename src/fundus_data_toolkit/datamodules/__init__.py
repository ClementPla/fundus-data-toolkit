import warnings
from enum import Enum
from typing import Dict

from fundus_data_toolkit.utils import usersettings
from fundus_data_toolkit.utils.collec import AttrDict

CLASSIF_PATHS = AttrDict()
SEG_PATHS = AttrDict()


USER_SETTING = usersettings.Settings("fundus_data_toolkit")
try:
    USER_SETTING.load_settings()
except usersettings.NoSectionError:
    USER_SETTING = None


if USER_SETTING:
    for key, value in USER_SETTING.items():
        if "classification" in key:
            CLASSIF_PATHS[key.split("classification_")[-1].upper()] = value
        elif "segmentation" in key:
            SEG_PATHS[key.split("segmentation_")[-1].upper()] = value
else:
    warnings.warn(
        "No settings found, please run `register_paths` to set paths or don't use CLASSIF_PATHS or SEG_PATHS."
    )


class Task(Enum):
    CLASSIFICATION: str = "classification"
    SEGMENTATION: str = "segmentation"

    @classmethod
    def _missing_(cls, value):
        value = value.lower()
        for member in cls:
            if member.lower() == value:
                return member
        return None


def register_paths(paths: Dict[str, str], task=Task.CLASSIFICATION):
    global CLASSIF_PATHS, SEG_PATHS
    setting = usersettings.Settings("fundus_data_toolkit")
    task = Task(task)
    for key, value in paths.items():
        match task:
            case Task.CLASSIFICATION:
                s = f"classification_{key}".lower()
                setting.add_setting(s, str, value)
                setting[s] = value
                CLASSIF_PATHS[key.upper()] = value
            case Task.SEGMENTATION:
                s = f"segmentation_{key}".lower()
                setting.add_setting(s, str, value)
                setting[s] = value
                SEG_PATHS[key.upper()] = value
            case _:
                raise ValueError(f"Task must be either {Task.CLASSIFICATION} or {Task.SEGMENTATION}, but got {task}")

    # Make sure we keep the old paths if they have not been overwritten
    for key, value in SEG_PATHS.items():
        if key not in paths:
            s = f"segmentation_{key}".lower()
            setting.add_setting(s, str, value)
            setting[s] = value
    for key, value in CLASSIF_PATHS.items():
        if key not in paths:
            s = f"classification_{key}".lower()
            setting.add_setting(s, str, value)
            setting[s] = value

    setting.save_settings()
