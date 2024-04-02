from enum import Enum


class DatasetVariant(Enum):
    TRAIN: str = "train"
    TEST: str = "test"
    VALID: str = "valid"
