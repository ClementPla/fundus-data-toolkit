from enum import Enum


class DatasetVariant(str, Enum):
    TRAIN: str = "train"
    TEST: str = "test"
    VALID: str = "valid"


class TJDR_COLORS(Enum):
    BG = (0, 0, 0)
    CWS = (0, 0, 128)
    EX = (128, 0, 0)
    HEM = (0, 128, 0)
    MA = (128, 128, 0)


class LesionIndex(Enum):
    BG = 0
    CWS = 1
    EX = 2
    HEM = 3
    MA = 4


class ODMAcIndex(Enum):
    BG = 0
    DISK = 1
    MACULA = 2


TJDR_color_interpretation = {
    TJDR_COLORS.BG.value: LesionIndex.BG.value,
    TJDR_COLORS.CWS.value: LesionIndex.CWS.value,
    TJDR_COLORS.EX.value: LesionIndex.EX.value,
    TJDR_COLORS.HEM.value: LesionIndex.HEM.value,
    TJDR_COLORS.MA.value: LesionIndex.MA.value,
}
