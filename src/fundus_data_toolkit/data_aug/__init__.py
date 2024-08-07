from enum import Enum


class DAType(Enum):
    DEFAULT = "default"
    AUTOAUGMENT = "autoaugment"
    RANDAUGMENT = "randaugment"
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"
    SUPERHEAVY = "superheavy"
    NONE = None

    @classmethod
    def _missing_(cls, value):
        value = value.lower()
        for member in cls:
            if member.lower() == value:
                return member
        return None
