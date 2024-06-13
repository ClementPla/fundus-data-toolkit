from enum import Enum, auto


class DAType(Enum):
    DEFAULT = "default"
    AUTOAUGMENT = "autoaugment"
    RANDAUGMENT = "randaugment"
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"
    NONE = None
    
    @classmethod
    def _missing_(cls, value):
        value = value.lower()
        for member in cls:
            if member.lower() == value:
                return member
        return None
