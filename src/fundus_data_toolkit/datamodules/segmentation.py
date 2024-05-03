from enum import Enum
from typing import List, Optional, Union

import albumentations as A
import numpy as np
from nntools.dataset.composer import CacheBullet, Composition, nntools_wrapper

from fundus_data_toolkit.data_aug import DAType
from fundus_data_toolkit.data_aug.segmentation import SegmentationDA
from fundus_data_toolkit.datamodules.common import FundusDatamodule
from fundus_data_toolkit.datasets.segmentation import (
    get_DDR_dataset,
    get_FGADR_dataset,
    get_IDRiD_dataset,
    get_MESSIDOR_dataset,
    get_RETLES_dataset,
)
from fundus_data_toolkit.datasets.utils import DatasetVariant
from fundus_data_toolkit.utils.image_processing import fundus_autocrop, fundus_precise_autocrop, image_check


class SegmentationType(Enum):
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"


@nntools_wrapper
def process_masks_multiclass(
    Exudates=None,
    Microaneurysms=None,
    Hemorrhages=None,
    Cotton_Wool_Spot=None,
    PreretinalHemorrhages=None,
    RetinalHemorrhages=None,
    VitreousHemorrhages=None,
    FibrousProliferation=None,
    IRMA=None,
    Neovascularization=None,
):
    if PreretinalHemorrhages is not None and RetinalHemorrhages is not None and VitreousHemorrhages is not None:
        Hemorrhages = np.logical_or(PreretinalHemorrhages, RetinalHemorrhages)
        Hemorrhages = np.logical_or(Hemorrhages, VitreousHemorrhages)

    mask = np.zeros_like(Exudates, dtype=np.uint8)
    mask[Cotton_Wool_Spot != 0] = 1
    mask[Exudates != 0] = 2
    mask[Hemorrhages != 0] = 3
    mask[Microaneurysms != 0] = 4
    return {"mask": mask}


@nntools_wrapper
def process_masks_multilabel(
    Exudates=None,
    Microaneurysms=None,
    Hemorrhages=None,
    Cotton_Wool_Spot=None,
    PreretinalHemorrhages=None,
    RetinalHemorrhages=None,
    VitreousHemorrhages=None,
    FibrousProliferation=None,
    IRMA=None,
    Neovascularization=None,
):
    if PreretinalHemorrhages is not None and RetinalHemorrhages is not None and VitreousHemorrhages is not None:
        Hemorrhages = np.logical_or(PreretinalHemorrhages, RetinalHemorrhages)
        Hemorrhages = np.logical_or(Hemorrhages, VitreousHemorrhages)

    mask = np.stack([Cotton_Wool_Spot, Exudates, Hemorrhages, Microaneurysms], axis=-1)
    return {"mask": mask}


class FundusSegmentationDatamodule(FundusDatamodule):
    def __init__(
        self,
        data_dir,
        img_size: Union[int, List[int]],
        batch_size: int,
        valid_size: Optional[Union[int, float]] = None,
        num_workers: int = 4,
        use_cache: bool = False,
        persistent_workers: bool = True,
        segmentation_type: Union[
            SegmentationType.MULTILABEL, SegmentationType.MULTICLASS
        ] = SegmentationType.MULTICLASS,
        data_augmentation_type: Optional[DAType] = None,
        random_crop: Optional[tuple[int, int]] = None,
        **dataset_kwargs,
    ):
        super().__init__(
            img_size,
            batch_size,
            valid_size,
            num_workers,
            use_cache,
            persistent_workers,
            data_augmentation_type=data_augmentation_type,
            **dataset_kwargs,
        )
        self.data_dir = data_dir
        self.seg_type = SegmentationType(segmentation_type)
        self.random_crop = random_crop

    def setup(self, stage: str):
        if stage == "validate":
            if self.train is None:
                self.setup("fit")
            self.create_valid_set()
        self.finalize_composition()

    def finalize_composition(self):
        test_composer = Composition()
        train_composer = Composition()
        autocrop = fundus_precise_autocrop if self.precise_autocrop else fundus_autocrop
        match self.seg_type:
            case SegmentationType.MULTILABEL:
                process_gt = process_masks_multilabel
            case SegmentationType.MULTICLASS:
                process_gt = process_masks_multiclass
        
        randomcrop = []
        if self.random_crop is not None:
            if isinstance(self.random_crop, int):
                self.random_crop = (self.random_crop, self.random_crop)
            randomcrop = [A.Compose([A.RandomCrop(*self.random_crop)], additional_targets={'roi': 'mask'})]
            
        test_composer.add(
            process_gt,
            autocrop,
            *self.pre_resize,
            self.img_size_ops(),
            *self.post_resize_pre_cache,
            CacheBullet(),
            *self.post_resize_post_cache,
            image_check,
            self.normalize_and_cast_op(),
        )

        train_composer = Composition()
        train_composer.add(
            process_gt,
            autocrop,
            *self.pre_resize,
            self.img_size_ops(),
            *self.post_resize_post_cache,
            CacheBullet(),
            *self.post_resize_post_cache,
            *self.data_aug_ops(),
            image_check,
            *randomcrop,
            self.normalize_and_cast_op(),
        )
        if self.train:
            self.train.composer = train_composer
        if self.val:
            self.val.composer = test_composer
        if self.test:
            self.test.composer = test_composer
    def data_aug_ops(self) -> Union[List[Composition], List[None]]:
        if self.da_type is None:
            return []
        return [SegmentationDA(self.da_type)]

class IDRiDDataModule_s(FundusSegmentationDatamodule):
    def setup(self, stage: str):
        match stage:
            case "fit" | "validate":
                self.train = get_IDRiD_dataset(
                    self.data_dir, DatasetVariant.TRAIN, self.img_size, **self.dataset_kwargs
                )
            case "test":
                self.test = get_IDRiD_dataset(self.data_dir, DatasetVariant.TEST, self.img_size, **self.dataset_kwargs)

        super().setup(stage)


class DDRDataModule_s(FundusSegmentationDatamodule):
    def setup(self, stage: str):
        match stage:
            case "fit":
                self.train = get_DDR_dataset(self.data_dir, DatasetVariant.TRAIN, self.img_size, **self.dataset_kwargs)
            case "validate":
                self.val = get_DDR_dataset(self.data_dir, DatasetVariant.VALID, self.img_size, **self.dataset_kwargs)
            case "test":
                self.test = get_DDR_dataset(self.data_dir, DatasetVariant.TEST, self.img_size, **self.dataset_kwargs)
        super().setup(stage)


class FGADRDataModule_s(FundusSegmentationDatamodule):
    def setup(self, stage: str):
        match stage:
            case "fit" | "validate":
                self.train = get_FGADR_dataset(
                    self.data_dir, DatasetVariant.TRAIN, self.img_size, **self.dataset_kwargs
                )
            case "test":
                self.test = get_FGADR_dataset(self.data_dir, DatasetVariant.TEST, self.img_size, **self.dataset_kwargs)
        super().setup(stage)


class MESSIDORDataModule_s(FundusSegmentationDatamodule):
    def setup(self, stage: str):
        match stage:
            case "fit" | "validate":
                self.train = get_MESSIDOR_dataset(
                    self.data_dir, DatasetVariant.TRAIN, self.img_size, **self.dataset_kwargs
                )
            case "test":
                self.test = get_MESSIDOR_dataset(
                    self.data_dir, DatasetVariant.TEST, self.img_size, **self.dataset_kwargs
                )
        super().setup(stage)


class RETLESDataModule_s(FundusSegmentationDatamodule):
    def setup(self, stage: str):
        match stage:
            case "fit" | "validate":
                self.train = get_RETLES_dataset(
                    self.data_dir, DatasetVariant.TRAIN, self.img_size, **self.dataset_kwargs
                )
            case "test":
                self.test = get_RETLES_dataset(self.data_dir, DatasetVariant.TEST, self.img_size, **self.dataset_kwargs)
        super().setup(stage)


if __name__ == '__main__':
    paths = {
    "ddr": "/home/clement/Documents/data/DDR/DDR-dataset/lesion_segmentation/",
    "fgadr": "/home/clement/Documents/data/FGADR/",
    "messidor": "/home/clement/Documents/data/Maples-DR/",
    "idrid": "/home/clement/Documents/data/IDRID/",
    "retles": "/home/clement/Documents/data/retinal-lesions-v20191227/",
    }
    
    ddr_datamodule = DDRDataModule_s(paths["ddr"], img_size=(512, 512), batch_size=1, 
                                 precise_autocrop=False,
                                 use_cache=False).setup_all()


    # retles_datamodule = RETLESDataModule_s(paths["retles"], img_size=(512, 512), batch_size=1, use_cache=False).setup_all()
    # retles_datamodule.train.plot(0)
    # import matplotlib.pyplot as plt
    # plt.show(block=True)