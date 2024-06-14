from typing import List, Union

from nntools.dataset.composer import CacheBullet, Composition

from fundus_data_toolkit.data_aug.classification import ClassificationDA
from fundus_data_toolkit.datamodules.common import FundusDatamodule
from fundus_data_toolkit.datasets.classification import (
    DatasetVariant,
    get_Aptos_dataset,
    get_DDR_dataset,
    get_EyePACS_dataset,
    get_IDRiD_dataset,
)
from fundus_data_toolkit.utils.image_processing import fundus_autocrop, fundus_precise_autocrop, image_check


class FundusClassificationDatamodule(FundusDatamodule):
    def __init__(
        self,
        data_dir,
        img_size,
        batch_size,
        valid_size=None,
        num_workers=4,
        use_cache=False,
        persistent_workers=True,
        filter_classes=None,
        precise_autocrop: bool = False,
        data_augmentation_type=None,
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
            precise_autocrop=precise_autocrop,
            **dataset_kwargs,
        )

        self.data_dir = data_dir
        self.filter_classes = filter_classes

    def data_aug_ops(self) -> Union[List[Composition], List[None]]:
        if self.da_type is None:
            return []
        return [ClassificationDA(self.da_type)]

    def finalize_composition(self):
        test_composer = Composition()
        train_composer = Composition()

        if not self.skip_autocrop:
            autocrop = fundus_precise_autocrop if self.precise_autocrop else fundus_autocrop
            train_composer.add(autocrop)
            test_composer.add(autocrop)

        test_composer.add(
            autocrop,
            *self.pre_resize,
            self.img_size_ops(),
            *self.post_resize_pre_cache,
            CacheBullet(),
            *self.post_resize_post_cache,
            image_check,
            self.normalize_and_cast_op(),
        )
        train_composer.add(
            autocrop,
            *self.pre_resize,
            self.img_size_ops(),
            *self.post_resize_post_cache,
            CacheBullet(),
            *self.post_resize_post_cache,
            *self.data_aug_ops(),
            image_check,
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

    def setup(self, stage: str):
        if stage == "validate":
            if self.train is None:
                self.setup("fit")
            self.create_valid_set()

        self.set_classes_filter()
        self.finalize_composition()


########### Data Modules for specific datasets ############


class DDRDataModule(FundusClassificationDatamodule):
    def __init__(
        self,
        data_dir,
        img_size,
        batch_size,
        valid_size=None,
        num_workers=4,
        use_cache=False,
        persistent_workers=True,
        filter_classes=5,
        **dataset_kwargs,
    ):
        super().__init__(
            data_dir,
            img_size,
            batch_size,
            valid_size,
            num_workers,
            use_cache,
            persistent_workers,
            filter_classes=filter_classes,
            **dataset_kwargs,
        )

    def setup(self, stage: str):
        if stage == "fit":
            self.train = get_DDR_dataset(self.data_dir, DatasetVariant.TRAIN, self.img_size, **self.dataset_kwargs)
        if stage == "validate":
            self.val = get_DDR_dataset(self.data_dir, DatasetVariant.VALID, self.img_size, **self.dataset_kwargs)
        if stage == "test":
            self.test = get_DDR_dataset(self.data_dir, DatasetVariant.TEST, self.img_size, **self.dataset_kwargs)
        super().setup(stage)


class IDRiDDataModule(FundusClassificationDatamodule):
    def setup(self, stage: str):
        if stage in ["fit", "validate"]:
            self.train = get_IDRiD_dataset(self.data_dir, DatasetVariant.TRAIN, self.img_size, **self.dataset_kwargs)
        if stage == "test":
            self.test = get_IDRiD_dataset(self.data_dir, DatasetVariant.TEST, self.img_size, **self.dataset_kwargs)
        super().setup(stage)


class EyePACSDataModule(FundusClassificationDatamodule):
    def setup(self, stage: str) -> None:
        if stage in ["fit", "validate"]:
            self.train = get_EyePACS_dataset(self.data_dir, DatasetVariant.TRAIN, self.img_size, **self.dataset_kwargs)
        if stage == "test":
            self.test = get_EyePACS_dataset(self.data_dir, DatasetVariant.TEST, self.img_size, **self.dataset_kwargs)
        super().setup(stage)


class AptosDataModule(FundusClassificationDatamodule):
    def setup(self, stage: str) -> None:
        if stage in ["fit", "validate"]:
            self.train = get_Aptos_dataset(self.data_dir, DatasetVariant.TRAIN, self.img_size, **self.dataset_kwargs)
        super().setup(stage)
