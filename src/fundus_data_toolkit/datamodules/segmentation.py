from abc import abstractmethod
from enum import Enum
from typing import List, Optional, Union

import albumentations as A
import numpy as np
from nntools.dataset.composer import CacheBullet, Composition, nntools_wrapper

from fundus_data_toolkit.data_aug import DAType
from fundus_data_toolkit.data_aug.segmentation import SegmentationDA
from fundus_data_toolkit.datamodules import DataHookPosition
from fundus_data_toolkit.datamodules.common import FundusDatamodule
from fundus_data_toolkit.datasets.segmentation import (
    get_APTOS_ODMAC_dataset,
    get_DDR_dataset,
    get_DDR_ODMAC_dataset,
    get_EYEPACS_ODMAC_dataset,
    get_FGADR_dataset,
    get_IDRiD_dataset,
    get_IDRID_ODMAC_dataset,
    get_MESSIDOR_dataset,
    get_RETLES_dataset,
    get_TJDR_dataset,
)
from fundus_data_toolkit.datasets.utils import DatasetVariant, LesionIndex, ODMAcIndex
from fundus_data_toolkit.utils.image_processing import (
    fundus_autocrop,
    fundus_precise_autocrop,
    fundus_roi,
    image_check,
)


class SegmentationType(str, Enum):
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
    Vessels=None,
    OpticDisc=None,
    OpticCup=None,
):
    if PreretinalHemorrhages is not None and RetinalHemorrhages is not None and VitreousHemorrhages is not None:
        Hemorrhages = np.logical_or(PreretinalHemorrhages, RetinalHemorrhages)
        Hemorrhages = np.logical_or(Hemorrhages, VitreousHemorrhages)

    mask = np.zeros_like(Exudates, dtype=np.uint8)
    if OpticDisc is not None:
        mask[OpticDisc != 0] = ODMAcIndex.DISK.value + 6
    if OpticCup is not None:
        mask[OpticCup != 0] = ODMAcIndex.MACULA.value + 6
    if Vessels is not None:
        mask[Vessels != 0] = 9

    mask[Cotton_Wool_Spot != 0] = LesionIndex.CWS.value
    mask[Exudates != 0] = LesionIndex.EX.value
    mask[Hemorrhages != 0] = LesionIndex.HEM.value
    mask[Microaneurysms != 0] = LesionIndex.MA.value
    if IRMA:
        mask[IRMA != 0] = LesionIndex.IRMA.value
    if Neovascularization:
        mask[Neovascularization != 0] = LesionIndex.NEO.value

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

    h, w = Exudates.shape
    mask = np.zeros((h, w, 4), dtype=np.uint8)
    mask[:, :, LesionIndex.CWS.value] = Cotton_Wool_Spot
    mask[:, :, LesionIndex.EX.value] = Exudates
    mask[:, :, LesionIndex.HEM.value] = Hemorrhages
    mask[:, :, LesionIndex.MA.value] = Microaneurysms
    return {"mask": mask}


@nntools_wrapper
def odmac_process_masks_multiclass(Disk, Macula):
    mask = np.zeros_like(Disk, dtype=np.uint8)
    mask[Disk != 0] = ODMAcIndex.DISK.value
    mask[Macula != 0] = ODMAcIndex.MACULA.value
    return {"mask": mask}


@nntools_wrapper
def odmac_process_masks_multilabel(Disk, Macula):
    h, w = Disk.shape
    mask = np.zeros((h, w, 2), dtype=np.uint8)
    mask[:, :, ODMAcIndex.DISK.value] = Disk
    mask[:, :, ODMAcIndex.MACULA.value] = Macula
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
        data_augmentation_type: Optional[DAType] = None,
        random_crop: Optional[tuple[int, int]] = None,
        segmentation_type: str = SegmentationType.MULTICLASS,
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
        self.random_crop = random_crop
        self.seg_type = SegmentationType(segmentation_type)

        process_gt = self.get_gt_process_fn()

        if process_gt is not None:
            self.set_data_pipeline_hook(process_gt, position=DataHookPosition.PRE_RESIZE)

    def setup(self, stage: str):
        if stage == "validate":
            if self.train is None:
                self.setup("fit")
            self.create_valid_set()
        self.finalize_composition()

    def finalize_composition(self):
        test_composer = Composition()
        train_composer = Composition()

        train_composer << self.pre_resize
        test_composer << self.pre_resize

        if self.skip_autocrop:
            train_composer.add(fundus_roi)
            test_composer.add(fundus_roi)
        else:
            autocrop = fundus_precise_autocrop if self.precise_autocrop else fundus_autocrop
            train_composer.add(autocrop)
            test_composer.add(autocrop)

        randomcrop = []
        if self.random_crop is not None:
            if isinstance(self.random_crop, int):
                self.random_crop = (self.random_crop, self.random_crop)
            randomcrop = [
                A.Compose(
                    [A.RandomCrop(*self.random_crop)],
                    additional_targets={"roi": "mask"},
                )
            ]

        test_composer.add(
            self.img_size_ops(),
            self.post_resize_pre_cache,
            CacheBullet(),
            self.post_resize_post_cache,
            image_check,
            self.normalize_and_cast_op(),
        )

        train_composer.add(
            self.img_size_ops(),
            self.post_resize_pre_cache,
            CacheBullet(),
            self.post_resize_post_cache,
            *(self.data_aug_ops()),
            image_check,
            *randomcrop,
            self.normalize_and_cast_op(),
        )
        if self.train:
            self.train.composer = train_composer
        if self.val:
            self.val.composer = test_composer
        if self.test:
            if isinstance(self.test, list):
                for test_set in self.test:
                    test_set.composer = test_composer
            else:
                self.test.composer = test_composer

        super().finalize_composition()

    def data_aug_ops(self) -> Union[List[Composition], List[None]]:
        if self.da_type is None or self.da_type == DAType.NONE:
            return []
        return [SegmentationDA(self.da_type)]

    @abstractmethod
    def get_gt_process_fn(self):
        pass


class FundusLesionsDatamodule(FundusSegmentationDatamodule):
    def get_gt_process_fn(self):
        match self.seg_type:
            case SegmentationType.MULTILABEL:
                return process_masks_multilabel
            case SegmentationType.MULTICLASS:
                return process_masks_multiclass


class FundusODMacDatamodule(FundusSegmentationDatamodule):
    def get_gt_process_fn(self):
        match self.seg_type:
            case SegmentationType.MULTILABEL:
                return odmac_process_masks_multilabel
            case SegmentationType.MULTICLASS:
                return odmac_process_masks_multiclass


class IDRiDDataModule_s(FundusLesionsDatamodule):
    def setup(self, stage: str):
        match stage:
            case "fit" | "validate":
                self.train = get_IDRiD_dataset(
                    self.data_dir,
                    DatasetVariant.TRAIN,
                    self.img_size,
                    **self.dataset_kwargs,
                )
            case "test":
                self.test = get_IDRiD_dataset(
                    self.data_dir,
                    DatasetVariant.TEST,
                    self.img_size,
                    **self.dataset_kwargs,
                )

        super().setup(stage)


class DDRDataModule_s(FundusLesionsDatamodule):
    def setup(self, stage: str):
        match stage:
            case "fit":
                self.train = get_DDR_dataset(
                    self.data_dir,
                    DatasetVariant.TRAIN,
                    self.img_size,
                    **self.dataset_kwargs,
                )
            case "validate":
                self.val = get_DDR_dataset(
                    self.data_dir,
                    DatasetVariant.VALID,
                    self.img_size,
                    **self.dataset_kwargs,
                )
            case "test":
                self.test = get_DDR_dataset(
                    self.data_dir,
                    DatasetVariant.TEST,
                    self.img_size,
                    **self.dataset_kwargs,
                )
        super().setup(stage)


class FGADRDataModule_s(FundusLesionsDatamodule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_autocrop = True

    def setup(self, stage: str):
        match stage:
            case "fit" | "validate":
                self.train = get_FGADR_dataset(
                    self.data_dir,
                    DatasetVariant.TRAIN,
                    self.img_size,
                    **self.dataset_kwargs,
                )
            case "test":
                self.test = get_FGADR_dataset(
                    self.data_dir,
                    DatasetVariant.TEST,
                    self.img_size,
                    **self.dataset_kwargs,
                )
        super().setup(stage)


class MESSIDORDataModule_s(FundusLesionsDatamodule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_autocrop = True

    def setup(self, stage: str):
        match stage:
            case "fit" | "validate":
                self.train = get_MESSIDOR_dataset(
                    self.data_dir,
                    DatasetVariant.TRAIN,
                    self.img_size,
                    **self.dataset_kwargs,
                )
            case "test":
                self.test = get_MESSIDOR_dataset(
                    self.data_dir,
                    DatasetVariant.TEST,
                    self.img_size,
                    **self.dataset_kwargs,
                )
        super().setup(stage)


class RETLESDataModule_s(FundusLesionsDatamodule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_autocrop = True

    def setup(self, stage: str):
        match stage:
            case "fit" | "validate":
                self.train = get_RETLES_dataset(
                    self.data_dir,
                    DatasetVariant.TRAIN,
                    self.img_size,
                    **self.dataset_kwargs,
                )
            case "test":
                self.test = get_RETLES_dataset(
                    self.data_dir,
                    DatasetVariant.TEST,
                    self.img_size,
                    **self.dataset_kwargs,
                )
        super().setup(stage)


class TJDRDataModule_s(FundusLesionsDatamodule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_autocrop = True

    def setup(self, stage: str):
        match stage:
            case "fit" | "validate":
                self.train = get_TJDR_dataset(
                    self.data_dir,
                    DatasetVariant.TRAIN,
                    self.img_size,
                    **self.dataset_kwargs,
                )
            case "test":
                self.test = get_TJDR_dataset(
                    self.data_dir,
                    DatasetVariant.TEST,
                    self.img_size,
                    **self.dataset_kwargs,
                )
        super().setup(stage)

    def get_gt_process_fn(self):
        return None


class MAPLESDRDataModule_s(FundusLesionsDatamodule):
    def setup(self, stage: str):
        match stage:
            case "fit" | "validate":
                self.train = get_MESSIDOR_dataset(
                    self.data_dir,
                    DatasetVariant.TRAIN,
                    self.img_size,
                    only_lesions=False,
                    **self.dataset_kwargs,
                )
            case "test":
                self.test = get_MESSIDOR_dataset(
                    self.data_dir,
                    DatasetVariant.TEST,
                    self.img_size,
                    only_lesions=False,
                    **self.dataset_kwargs,
                )
        super().setup(stage)


class IDRIDODMACDataModule(FundusODMacDatamodule):
    def setup(self, stage: str):
        match stage:
            case "fit" | "validate":
                self.train = get_IDRID_ODMAC_dataset(
                    self.data_dir,
                    DatasetVariant.TRAIN,
                    self.img_size,
                    **self.dataset_kwargs,
                )
            case "test":
                self.test = get_IDRID_ODMAC_dataset(
                    self.data_dir,
                    DatasetVariant.TEST,
                    self.img_size,
                    **self.dataset_kwargs,
                )

        super().setup(stage)


class EYEPACSODMACDataModule(FundusODMacDatamodule):
    def setup(self, stage: str):
        match stage:
            case "fit" | "validate":
                self.train = get_EYEPACS_ODMAC_dataset(
                    self.data_dir,
                    DatasetVariant.TRAIN,
                    self.img_size,
                    **self.dataset_kwargs,
                )
            case "test":
                self.test = get_EYEPACS_ODMAC_dataset(
                    self.data_dir,
                    DatasetVariant.TEST,
                    self.img_size,
                    **self.dataset_kwargs,
                )

        super().setup(stage)


class DDRODMACDataModule(FundusODMacDatamodule):
    def setup(self, stage: str):
        match stage:
            case "fit" | "validate":
                self.train = get_DDR_ODMAC_dataset(
                    self.data_dir,
                    DatasetVariant.TRAIN,
                    self.img_size,
                    **self.dataset_kwargs,
                )
            case "test":
                self.test = get_DDR_ODMAC_dataset(
                    self.data_dir,
                    DatasetVariant.TEST,
                    self.img_size,
                    **self.dataset_kwargs,
                )

        super().setup(stage)


class APTOSODMACDataModule(FundusODMacDatamodule):
    def setup(self, stage: str):
        match stage:
            case "fit" | "validate":
                self.train = get_APTOS_ODMAC_dataset(
                    self.data_dir,
                    DatasetVariant.TRAIN,
                    self.img_size,
                    **self.dataset_kwargs,
                )
            case "test":
                self.test = get_APTOS_ODMAC_dataset(
                    self.data_dir,
                    DatasetVariant.TEST,
                    self.img_size,
                    **self.dataset_kwargs,
                )

        super().setup(stage)
