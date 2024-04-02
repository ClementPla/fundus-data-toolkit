
import albumentations as A
import cv2

from fundusData.data_aug import DAType


class ClassificationDA:
    def __init__(self, type:DAType = DAType.DEFAULT):
        self.type = type
    
    def get_data_aug(self):
        if self.type == DAType.DEFAULT:
            return ClassificationDA.default_transform()
        if self.type == DAType.AUTOAUGMENT:
            return ClassificationDA.autoaugment_transform()
        if self.type == DAType.RANDAUGMENT:
            return ClassificationDA.randaugment_transform()
        if self.type == DAType.LIGHT:
            return ClassificationDA.light_transform()
        if self.type == DAType.MEDIUM:
            return ClassificationDA.medium_transform()
        if self.type == DAType.HEAVY:
            return ClassificationDA.heavy_transform()

    @staticmethod
    def light_transform() -> A.Compose:
        return A.Compose([A.OneOf([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)]),
            A.ShiftScaleRotate(p=0.5, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT),
            ])

    @staticmethod
    def medium_transform() -> A.Compose:
        A.Compose(
                [   *ClassificationDA.light_transform(),
                    A.RandomBrightnessContrast(p=0.5),
                ]
            )
    
    @staticmethod
    def heavy_transform() -> A.Compose:
        A.Compose(
            [
                *ClassificationDA.medium_transform(),
                A.HueSaturationValue(p=0.5),
                A.Blur(blur_limit=3, p=0.1),
            ]
        )
    
    @staticmethod
    def default_transform() -> A.Compose:
        return ClassificationDA.medium_transform()

    @staticmethod
    def autoaugment_transform(n:int=7, m:int=5) -> A.Compose:
        raise NotImplementedError
    
    @staticmethod
    def randaugment_transform() -> A.Compose:
        raise NotImplementedError