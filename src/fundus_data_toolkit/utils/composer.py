import albumentations as A
import cv2
from albumentations.pytorch.transforms import ToTensorV2
from fundus_data_toolkit.config import get_normalization
from fundus_data_toolkit.utils.image_processing import fundus_autocrop, fundus_precise_autocrop
from nntools.dataset.composer import Composition


def get_generic_composer(shape, precise: bool = False):
    mean, std = get_normalization()
    composer = Composition()
    
    if precise:
        composer.add(fundus_precise_autocrop)
    else:
        composer.add(fundus_autocrop)
        
    
    resize_op = A.Compose([
            A.LongestMaxSize(max_size=shape, always_apply=True),
            A.PadIfNeeded(
                min_height=shape[0],
                min_width=shape[1],
                always_apply=True,
                border_mode=cv2.BORDER_CONSTANT,
            )
        ])

    composer << resize_op << A.Normalize(mean=mean, std=std, always_apply=True) << ToTensorV2()
    return composer