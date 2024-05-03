import os
from abc import abstractmethod
from typing import TYPE_CHECKING, List, Optional, Union

import albumentations as A
import cv2
import torch
from albumentations.pytorch.transforms import ToTensorV2
from nntools.dataset.utils.balance import class_weighting
from nntools.dataset.utils.ops import random_split
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from fundus_data_toolkit.config import get_normalization
from fundus_data_toolkit.data_aug import DAType

if TYPE_CHECKING:
    import nntools.dataset as D


class FundusDatamodule(LightningDataModule):
    def __init__(
        self,
        img_size: Union[int, List[int]],
        batch_size: int,
        valid_size: Optional[Union[int, float]] = None,
        num_workers: int = 4,
        use_cache: bool = False,
        persistent_workers: bool = True,
        precise_autocrop: bool = False,
        eval_batch_size: Optional[int] = None,
        data_augmentation_type: Optional[DAType] = None,
        **dataset_kwargs,
    ):
        super().__init__()
        self.img_size = img_size
        self.valid_size = valid_size
        
        self.batch_size = batch_size // torch.cuda.device_count()
        
        if eval_batch_size is None:
            self.eval_batch_size = batch_size
        else:
            self.eval_batch_size = eval_batch_size
            
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.da_type = data_augmentation_type
        self.precise_autocrop = precise_autocrop

        if num_workers == "auto":
            self.num_workers = os.cpu_count() // torch.cuda.device_count()
        else:
            self.num_workers = num_workers

        self.train: Union[D.ClassificationDataset, D.SegmentationDataset] = None
        self.val: Union[D.ClassificationDataset, D.SegmentationDataset] = None
        self.test: Union[D.ClassificationDataset, D.SegmentationDataset] = None
        self.dataset_kwargs = dataset_kwargs
        self.dataset_kwargs["use_cache"] = use_cache

        self.pre_resize = []
        self.post_resize_pre_cache = []
        self.post_resize_post_cache = []

    def setup_all(self):
        self.setup("fit")
        self.setup("validate")
        self.setup("test")
        return self

    @abstractmethod
    def finalize_composition(self):
        pass

    def set_classes_filter(self):
        if self.filter_classes is not None:
            if self.train:
                self.train.filter_classes(self.filter_classes)
            if self.val:
                self.val.filter_classes(self.filter_classes)
            if self.test:
                self.test.filter_classes(self.filter_classes)

    def create_valid_set(self):
        if self.train and self.val is None and self.valid_size:
            if isinstance(self.valid_size, float):
                self.valid_size = int(len(self.train) * self.valid_size)

            val_length = self.valid_size
            train_length = len(self.train) - val_length
            self.train, self.val = random_split(
                self.train, [train_length, val_length], generator=torch.Generator().manual_seed(42)
            )
            self.train.id = self.train.id.replace("_split_0", "_split_train")
            self.val.id = self.val.id.replace("_split_1", "_split_val")

    @property
    def class_weights(self) -> torch.Tensor:
        if self.train is None:
            raise ValueError("Train dataset is not created yet.")

        return torch.Tensor(class_weighting(self.train.get_class_count()))

    @property
    def class_count(self) -> List[int]:
        if self.train is None:
            raise ValueError("Train dataset is not created yet.")

        return self.train.get_class_count()

    def img_size_ops(self) -> A.Compose:
        return A.Compose(
            [
                A.LongestMaxSize(max_size=self.img_size, always_apply=True),
                A.PadIfNeeded(
                    min_height=self.img_size[0],
                    min_width=self.img_size[1],
                    always_apply=True,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
            ],
            additional_targets={"roi": "mask"},
        )

    def normalize_and_cast_op(self):
        mean, std = get_normalization()
        return A.Compose(
            [A.Normalize(mean=mean, std=std, always_apply=True), ToTensorV2()], additional_targets={"roi": "mask"}
        )

    def train_dataloader(self) -> DataLoader:
        if self.train is None:
            raise ValueError("Train dataset is not created yet.")
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self, shuffle=True, persistent_workers=True) -> DataLoader:
        if self.val is None:
            raise ValueError("Valid dataset is not created yet.")
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers and persistent_workers and self.num_workers > 0,
            pin_memory=True,
        )

    def test_dataloader(self, shuffle=False) -> Union[DataLoader, List[DataLoader]]:
        if self.test is None:
            raise ValueError("Test dataset is not created yet.")

        if isinstance(self.test, list):
            return [
                DataLoader(
                    ds,
                    batch_size=self.eval_batch_size,
                    num_workers=self.num_workers,
                    shuffle=shuffle,
                    persistent_workers=False,
                    pin_memory=True,
                )
                for ds in self.test
            ]

        return DataLoader(
            self.test,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            persistent_workers=False,
            pin_memory=True,
        )
