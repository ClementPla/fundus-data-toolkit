from pathlib import Path
from typing import Tuple

import torch
from nntools import NNOpt
from nntools.dataset import SegmentationDataset, random_split

from fundus_data_toolkit.datasets.utils import DatasetVariant


def get_IDRiD_dataset(root: str, variant: DatasetVariant, img_size: Tuple[int, int], **kwargs) -> SegmentationDataset:
    def sort_func_idrid(x):
        return int(x.split("_")[1])

    root = Path(root)

    img_root = root / "1. Original Images/"
    mask_root = root / "2. All Segmentation Groundtruths/"

    match variant:
        case DatasetVariant.TRAIN:
            img_root = img_root / "a. Training Set/"
            mask_root = mask_root / "a. Training Set/"
        case DatasetVariant.TEST:
            img_root = img_root / "b. Testing Set/"
            mask_root = mask_root / "b. Testing Set/"
        case DatasetVariant.VALID:
            raise ValueError("No explicit validation set for IDRiD dataset")

    masks = {
        "Exudates": mask_root / "3. Hard Exudates/",
        "Cotton_Wool_Spot": mask_root / "4. Soft Exudates/",
        "Hemorrhages": mask_root / "2. Haemorrhages/",
        "Microaneurysms": mask_root / "1. Microaneurysms/",
    }

    dataset = SegmentationDataset(
        img_root=img_root,
        shape=img_size,
        keep_size_ratio=True,
        auto_pad=True, auto_resize=True,
        mask_root=masks,
        binarize_mask=True,
        extract_image_id_function=sort_func_idrid,
        filling_strategy=NNOpt.FILL_UPSAMPLE,
        id=f'IDRID_{variant.value}',
        **kwargs,
    )
    return dataset


def get_DDR_dataset(root: str, variant: DatasetVariant, img_size: Tuple[int, int], **kwargs) -> SegmentationDataset:
    root = Path(root)

    img_root = root / variant.value / "image/"
    
    mask_root = root / variant.value / "label/"
    if variant == DatasetVariant.VALID:
        mask_root = root / variant.value / "segmentation label/"
    masks = {
        "Exudates": mask_root / "EX/",
        "Cotton_Wool_Spot": mask_root / "SE/",
        "Hemorrhages": mask_root / "HE/",
        "Microaneurysms": mask_root / "MA/",
    }

    dataset = SegmentationDataset(
        img_root=img_root,
        shape=img_size,
        keep_size_ratio=True,
        auto_pad=True, auto_resize=True,
        mask_root=masks,
        binarize_mask=True,
        filling_strategy=NNOpt.FILL_UPSAMPLE,
        id=f'DDR_{variant.value}',
        **kwargs
    )
    return dataset

def get_MESSIDOR_dataset(root: str, variant: DatasetVariant, img_size: Tuple[int, int], **kwargs) -> SegmentationDataset:
    if variant == DatasetVariant.VALID:
        raise ValueError("No explicit validation set for MESSIDOR dataset")
    
    root = Path(root)

    img_root = root / variant.value / "fundus/"
    mask_root = root / variant.value

    masks = {
        "Exudates": mask_root / "exudates/",
        "Cotton_Wool_Spot": mask_root / "cottonWoolSpots/",
        "Hemorrhages": mask_root / "hemorrhages/",
        "Microaneurysms": mask_root / "microaneurysms/",
    }

    dataset = SegmentationDataset(
        img_root=img_root,
        shape=img_size,
        keep_size_ratio=True,
        auto_pad=True, auto_resize=True,
        mask_root=masks,
        binarize_mask=True,
        filling_strategy=NNOpt.FILL_UPSAMPLE,
        id=f'MESSIDOR_{variant.value}',
        **kwargs
    )
    return dataset

def get_MAPLESDR_dataset(root: str, variant: DatasetVariant, img_size: Tuple[int, int], **kwargs) -> SegmentationDataset:
    return get_MESSIDOR_dataset(root, variant, img_size, **kwargs)

def get_FGADR_dataset(root:str, variant:DatasetVariant, img_size: Tuple[int, int], **kwargs) -> SegmentationDataset:

    root = Path(root)

    img_root = root / "Original_Images"
    mask_root = root
    masks = {
        "Exudates": mask_root / "HardExudate_Masks",
        "Cotton_Wool_Spot": mask_root / "SoftExudate_Masks",
        "Hemorrhages": mask_root / "Hemohedge_Masks",
        "Microaneurysms": mask_root / "Microaneurysms_Masks",
        "IRMA": mask_root / "IRMA_Masks",
        "Neovascularization": mask_root / "Neovascularization_Masks",
    }

    dataset = SegmentationDataset(
        img_root=img_root,
        shape=img_size,
        keep_size_ratio=True,
        auto_pad=True, auto_resize=True,
        mask_root=masks,
        binarize_mask=True,
        filling_strategy=NNOpt.FILL_UPSAMPLE,
        id='FGADR',
        **kwargs
    )
    train, test = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
    match variant:
        case DatasetVariant.TRAIN:
            return train
        case DatasetVariant.TEST:
            return test
        case DatasetVariant.VALID:
            raise ValueError("No explicit validation set for FGADR dataset")

def get_RETLES_dataset(root: str, variant: DatasetVariant, img_size: Tuple[int, int], **kwargs) -> SegmentationDataset:
    root = Path(root)

    img_root = root / "images_896x896"
    mask_root = root / "segmentation"

    masks = {
        "Exudates": mask_root / "hard_exudate",
        "Cotton_Wool_Spot": mask_root / "cotton_wool_spots",
        "Microaneurysms": mask_root / "microaneurysm",
        "PreretinalHemorrhages": mask_root / "preretinal_hemorrhage",
        "RetinalHemorrhages": mask_root / "retinal_hemorrhage",
        "VitreousHemorrhages": mask_root / "vitreous_hemorrhage",
        "FibrousProliferation" : mask_root / "fibrous_proliferation",
        "Neovascularization" : mask_root / "neovascularization",
    }

    dataset = SegmentationDataset(
        img_root=img_root,
        shape=img_size,
        keep_size_ratio=True,
        auto_pad=True, auto_resize=True,
        mask_root=masks,
        binarize_mask=True,
        filling_strategy=NNOpt.FILL_UPSAMPLE,
        id='RETLES',
        **kwargs
    )
    train, test = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
    match variant:
        case DatasetVariant.TRAIN:
            return train
        case DatasetVariant.TEST:
            return test
        case DatasetVariant.VALID:
            raise ValueError("No explicit validation set for RETLES dataset")
