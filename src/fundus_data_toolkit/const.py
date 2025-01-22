from enum import Enum

DEFAULT_COLORS = ["black", "#eca63f", "#8cf18e", "#4498f0", "#141488"]


class DataHookPosition(Enum):
    PRE_RESIZE = "pre_resize"
    POST_RESIZE_PRE_CACHE = "post_resize_pre_cache"
    POST_RESIZE_POST_CACHE = "post_resize_post_cache"


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
