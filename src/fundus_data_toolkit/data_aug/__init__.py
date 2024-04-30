from enum import Enum, auto


class DAType(Enum):
    DEFAULT = auto()
    AUTOAUGMENT = auto()
    RANDAUGMENT = auto()
    LIGHT = auto()
    MEDIUM = auto()
    HEAVY = auto()