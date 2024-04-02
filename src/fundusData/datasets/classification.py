import os
from typing import Tuple

import pandas as pd
from nntools.dataset import ClassificationDataset

from fundusData.datasets.utils import DatasetVariant
from fundusData.utils.path_processing import filename_without_extension


def get_DDR_dataset(root:str, variant:DatasetVariant, img_size: Tuple[int, int]) -> ClassificationDataset:

    df = pd.read_csv(os.path.join(root, f"{variant.value}.txt"), sep=" ", names=["image", "label"])
    dataset = ClassificationDataset(
        os.path.join(root, f"{variant.value}/"),
        label_dataframe=df,
        shape=img_size,
        keep_size_ratio=True,
        use_cache=False,
        auto_pad=True,
    )
    return dataset


def get_IDRiD_dataset(root:str, variant:DatasetVariant, img_size: Tuple[int, int]) -> ClassificationDataset:
    
    match variant:
        case DatasetVariant.TRAIN:
            img_dir = os.path.join(root, "1. Original Images/a. Training Set/")
            label_filepath = os.path.join(root, "2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv")
        case DatasetVariant.TEST:
            label_filepath = os.path.join(root, "2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv")
            img_dir = os.path.join(root, "1. Original Images/b. Testing Set/")

    dataset = ClassificationDataset(
                img_dir,
                shape=img_size,
                keep_size_ratio=True,
                file_column="Image name",
                gt_column="Retinopathy grade",
                label_filepath=label_filepath,
            )
    
    dataset.remap("Retinopathy grade", "label")
    return dataset


def get_EyePACS_dataset(root:str, variant:DatasetVariant, img_size: Tuple[int, int]) -> ClassificationDataset:
    match variant:
        case DatasetVariant.TRAIN:
            img_dir = os.path.join(root, "train/images/")
            label_filepath = os.path.join(root, "trainLabels.csv")
        case DatasetVariant.TEST:
            img_dir = os.path.join(root, "test/images/")
            label_filepath = os.path.join(root, "testLabels.csv")

    dataset = ClassificationDataset(
        img_dir,
        label_filepath=label_filepath,
        file_column="image",
        gt_column="level",
        shape=img_size,
        keep_size_ratio=True,
        use_cache=False,
        auto_pad=True,
        extract_image_id_function=filename_without_extension
    )
    dataset.remap("level", "label")

    return dataset

def get_Aptos_dataset(root, variant:DatasetVariant, img_size: Tuple[int, int]) -> ClassificationDataset:
    dataset = ClassificationDataset(
        os.path.join(root, "train/"),
        label_filepath=os.path.join(root, "train.csv"),
        file_column="id_code",
        gt_column="diagnosis",
        shape=img_size,
        keep_size_ratio=True,
        auto_pad=True,
    )
    dataset.remap("diagnosis", "label")
    
    return dataset