from enum import Enum

from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)


class NormalizationConst(Enum):
    IMAGENET_DEFAULT = (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    IMAGENET_INCEPTION = (IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD)
    NO_NORMALIZATION = ([0, 0, 0], [1.0, 1.0, 1.0])


DEFAULT_NORMALIZATION = NormalizationConst.IMAGENET_DEFAULT


def setup_normalization(normalization: NormalizationConst):
    global DEFAULT_NORMALIZATION
    DEFAULT_NORMALIZATION = normalization


def get_normalization():
    return DEFAULT_NORMALIZATION.value
