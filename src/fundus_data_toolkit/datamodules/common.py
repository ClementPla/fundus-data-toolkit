import os
from abc import abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Union

import albumentations as A
import cv2
import torch
from albumentations.pytorch.transforms import ToTensorV2
from nntools.dataset.composer import Composition
from nntools.dataset.utils.balance import class_weighting
from nntools.dataset.utils.concat import concat_datasets_if_needed
from nntools.dataset.utils.ops import random_split
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from fundus_data_toolkit.config import get_normalization
from fundus_data_toolkit.data_aug import DAType

if TYPE_CHECKING:
    import nntools.dataset as D

from fundus_data_toolkit.datamodules import DataHookPosition


class BaseDatamodule(LightningDataModule):
    def __init__(
        self,
        img_size: Union[int, List[int]],
        batch_size: int,
        valid_size: Optional[Union[int, float]] = None,
        num_workers: int = 4,
        persistent_workers: bool = True,
        eval_batch_size: Optional[int] = None,
        drop_last: bool = False,
    ):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        self.img_size = img_size
        self.valid_size = valid_size

        self.batch_size = batch_size // max(1, torch.cuda.device_count())

        if eval_batch_size is None:
            self.eval_batch_size = batch_size
        else:
            self.eval_batch_size = eval_batch_size

        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.drop_last = drop_last

        if num_workers == "auto":
            self.num_workers = os.cpu_count() // torch.cuda.device_count()
        else:
            self.num_workers = num_workers
        self.train_shuffle = True

    def setup_all(self):
        self.setup("fit")
        self.setup("validate")
        self.setup("test")
        return self

    def set_data_pipeline_hook(self, *fns, position=None):
        assert position is not None, "Position must be specified"
        match position:
            case DataHookPosition.PRE_RESIZE:
                self.pre_resize.add(*fns)
            case DataHookPosition.POST_RESIZE_PRE_CACHE:
                self.post_resize_pre_cache.add(*fns)
            case DataHookPosition.POST_RESIZE_POST_CACHE:
                self.post_resize_post_cache.add(*fns)
            case _:
                raise ValueError(f"Invalid position: {position}")

    def return_tag(self, value):
        if self.train:
            if isinstance(self.train, list):
                for train_set in self.train:
                    train_set.return_tag = value
            else:
                self.train.return_tag = value
        if self.val:
            if isinstance(self.val, list):
                for val_set in self.val:
                    val_set.return_tag = value
            else:
                self.val.return_tag = value
        if self.test:
            if isinstance(self.test, list):
                for test_set in self.test:
                    test_set.return_tag = value
            else:
                self.test.return_tag = value

    @abstractmethod
    def finalize_composition(self):
        if self.test:
            if isinstance(self.test, list):
                for test_set in self.test:
                    test_set.return_indices = True
            else:
                self.test.return_indices = True

        if self.train and self.train.use_cache:
            self.train.init_cache()
        if self.val and self.val.use_cache:
            self.val.init_cache()

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

    def train_dataloader(self) -> DataLoader:
        if self.train is None:
            raise ValueError("Train dataset is not created yet.")
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
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
                    persistent_workers=True,
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

    def add_target(self, additional_target):
        assert (
            self.train is not None or self.val is not None or self.test is not None
        ), "No dataset is created yet, please call setup all."
        if self.train:
            for op in self.train.composer.ops:
                if isinstance(op["f"], A.Compose):
                    op["f"].add_targets(additional_target)

        if self.val:
            for op in self.val.composer.ops:
                if isinstance(op["f"], A.Compose):
                    op["f"].add_targets(additional_target)

        if self.test:
            for op in self.test.composer.ops:
                if isinstance(op["f"], A.Compose):
                    op["f"].add_targets(additional_target)


class FundusDatamodule(BaseDatamodule):
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
        skip_autocrop: bool = False,
        drop_last: bool = False,
        **dataset_kwargs,
    ):
        super().__init__(
            img_size=img_size,
            batch_size=batch_size,
            valid_size=valid_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            eval_batch_size=eval_batch_size,
            drop_last=drop_last,
        )
        self.train: Union[D.ClassificationDataset, D.SegmentationDataset] = None
        self.val: Union[D.ClassificationDataset, D.SegmentationDataset] = None
        self.test: Union[D.ClassificationDataset, D.SegmentationDataset] = None
        self.dataset_kwargs = dataset_kwargs
        self.dataset_kwargs["use_cache"] = use_cache
        self.pre_resize = Composition()
        self.post_resize_pre_cache = Composition()
        self.post_resize_post_cache = Composition()
        self.da_type = DAType(data_augmentation_type)
        self.precise_autocrop = precise_autocrop
        self.skip_autocrop = skip_autocrop

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

    def img_size_ops(self) -> A.Compose:
        return A.Compose(
            [
                A.LongestMaxSize(max_size=self.img_size, always_apply=True),
                A.PadIfNeeded(
                    min_height=self.img_size[0],
                    min_width=self.img_size[1],
                    always_apply=True,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
            ],
            strict=False,
            additional_targets={"roi": "mask"},
        )

    def normalize_and_cast_op(self):
        mean, std = get_normalization()
        return A.Compose(
            [A.Normalize(mean=mean, std=std, always_apply=True), ToTensorV2()],
            additional_targets={"roi": "mask"},
            strict=False,
        )


class MergedDatamodule(BaseDatamodule):
    def __init__(self, *datamodules: FundusDatamodule, separate_test_sets: bool = True):
        if len(datamodules) == 0:
            raise ValueError("No datamodules to merge.")

        if len(datamodules) == 1:
            return datamodules[0]

        assert all(
            isinstance(dm, FundusDatamodule) for dm in datamodules
        ), "All datamodules must be of type FundusDatamodule"

        img_size = set([tuple(dm.img_size) for dm in datamodules])
        num_workers = set([dm.num_workers for dm in datamodules])
        batch_size = set([dm.batch_size for dm in datamodules])

        assert len(img_size) == 1, "All datamodules must have the same img_size"
        assert len(num_workers) == 1, "All datamodules must have the same num_workers"
        assert len(batch_size) == 1, "All datamodules must have the same batch_size"

        super().__init__(img_size=datamodules[0].img_size, batch_size=datamodules[0].batch_size)
        self.datamodules = datamodules
        self.separate_test_sets = separate_test_sets

    def setup(self, stage: Optional[str] = None):
        for datamodule in self.datamodules:
            datamodule.setup(stage)

    def finalize_composition(self):
        for datamodule in self.datamodules:
            datamodule.finalize_composition()

    @property
    def train(self):
        return concat_datasets_if_needed([dm.train for dm in self.datamodules if dm.train is not None])

    @property
    def val(self):
        return concat_datasets_if_needed([dm.val for dm in self.datamodules if dm.val is not None])

    @property
    def test(self):
        if self.separate_test_sets:
            return [dm.test for dm in self.datamodules if dm.test is not None]
        return concat_datasets_if_needed([dm.test for dm in self.datamodules if dm.test is not None])

    def add_target(self, additional_target):
        for dm in self.datamodules:
            dm.add_target(additional_target)

    def set_data_pipeline_hook(self, *fns, position=None):
        for dm in self.datamodules:
            dm.set_data_pipeline_hook(*fns, position=position)

    def set_classes_filter(self):
        for dm in self.datamodules:
            dm.set_classes_filter()

    def return_tag(self, value):
        for dm in self.datamodules:
            dm.return_tag(value)
