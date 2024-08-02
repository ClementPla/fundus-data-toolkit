from abc import ABCMeta

import albumentations as A
import cv2

from fundus_data_toolkit.data_aug import DAType


class ClassificationDA(ABCMeta):
    def __new__(cls, type: DAType = DAType.DEFAULT):
        match type:
            case DAType.DEFAULT:
                return cls.default_transform()
            case DAType.AUTOAUGMENT:
                return cls.autoaugment_transform()
            case DAType.RANDAUGMENT:
                return cls.randaugment_transform()
            case DAType.LIGHT:
                return cls.light_transform()
            case DAType.MEDIUM:
                return cls.medium_transform()
            case DAType.HEAVY:
                return cls.heavy_transform()
            case DAType.SUPERHEAVY:
                return cls.superheavy_transform()
            case DAType.NONE:
                return None
            case _:
                return cls.default_transform()

    @staticmethod
    def light_transform() -> A.Compose:
        return A.Compose(
            [
                A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]),
                A.ShiftScaleRotate(p=0.5, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT),
            ],
            additional_targets={"roi": "mask"},
            strict=False,
        )

    @staticmethod
    def medium_transform() -> A.Compose:
        return A.Compose(
            [
                *ClassificationDA.light_transform(),
                A.RandomBrightnessContrast(p=0.5),
            ],
            additional_targets={"roi": "mask"},
            strict=False,
        )

    @staticmethod
    def heavy_transform() -> A.Compose:
        return A.Compose(
            [
                *ClassificationDA.medium_transform(),
                A.HueSaturationValue(p=0.5),
                A.Blur(blur_limit=3, p=0.1),
            ],
            additional_targets={"roi": "mask"},
            strict=False,
        )

    @staticmethod
    def superheavy_transform() -> A.Compose:
        return A.Compose(
            [
                A.OneOf(
                    [
                        A.HorizontalFlip(),
                        A.VerticalFlip(),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.ColorJitter(),
                        A.Equalize(),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.InvertImg(),
                        A.Rotate(),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.Posterize(),
                        A.Solarize(),
                        A.ColorJitter(),
                    ],
                    p=0.5,
                ),
                A.Sharpen(),
                A.Affine(shear=[-5, 5]),
                A.RandomBrightnessContrast(),
            ],
            additional_targets={"roi": "mask"},
            strict=False,
        )

    @staticmethod
    def default_transform() -> A.Compose:
        return ClassificationDA.medium_transform()

    @staticmethod
    def autoaugment_transform(n: int = 7, m: int = 5) -> A.Compose:
        raise NotImplementedError

    @staticmethod
    def randaugment_transform() -> A.Compose:
        raise NotImplementedError
