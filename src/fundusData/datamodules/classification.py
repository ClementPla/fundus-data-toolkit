from typing import List, Union

from nntools.dataset import Composition

from fundusData.data_aug.classification import ClassificationDA
from fundusData.datamodules.common import FundusDatamodule
from fundusData.datasets.classification import (
    DatasetVariant,
    get_Aptos_dataset,
    get_DDR_dataset,
    get_EyePACS_dataset,
    get_IDRiD_dataset,
)
from fundusData.utils.image_processing import fundus_autocrop


class FundusClassificationDatamodule(FundusDatamodule):
    def __init__(self, data_dir, img_size, batch_size, 
                 valid_size=None, 
                 num_workers=4, 
                 use_cache=False, 
                 persistent_workers=True, filter_classes=None):
        super().__init__(img_size, batch_size, valid_size, num_workers, use_cache, persistent_workers)

        self.data_dir = data_dir
        self.filter_classes = filter_classes

    
    def data_aug_ops(self) -> Union[List[Composition], List[None]]:
        if self.da_type is None:
            return []
        return [ClassificationDA(self.da_type).get_data_aug()]
    
    def finalize_composition(self):
        test_composer = Composition()
        test_composer.add(fundus_autocrop, self.img_size_ops(), self.normalize_and_cast_op())
        train_composer = Composition()
        train_composer.add(fundus_autocrop, self.img_size_ops(), *self.data_aug_ops(), self.normalize_and_cast_op())
        if self.train:
            self.train.composer = train_composer
        if self.val:
            self.val.composer = test_composer
        if self.test:
            self.test.composer = test_composer

    def setup(self, stage: str):
        if stage == "validate":
            if self.train is None:
                self.setup("fit")

            self.create_valid_set()

        self.set_classes_filter()
        self.finalize_composition()
    
    

########### Data Modules for specific datasets ############

class DDRDataModule(FundusClassificationDatamodule):
    def __init__(self, data_dir, img_size,  
                 batch_size, valid_size=None, num_workers=4, 
                 use_cache=False, 
                 persistent_workers=True, 
                 filter_classes=5):
        super().__init__(data_dir, img_size, batch_size, valid_size, 
                         num_workers, 
                         use_cache, 
                         persistent_workers, 
                         filter_classes=filter_classes)
    
    def setup(self, stage: str):
        if stage == "fit":
            self.train = get_DDR_dataset(self.data_dir, DatasetVariant.TRAIN, self.img_size)
        if stage == "validate":
            self.val = get_DDR_dataset(self.data_dir, DatasetVariant.VALID, self.img_size)
        if stage == "test":
            self.test = get_DDR_dataset(self.data_dir, 
                                        DatasetVariant.TEST, self.img_size)
        super().setup(stage)


class IDRiDDataModule(FundusClassificationDatamodule):
    def setup(self, stage: str):
        if stage in ["fit", "validate"]:
            self.train =  get_IDRiD_dataset(self.data_dir, DatasetVariant.TRAIN, self.img_size)
        if stage == "test":
            self.test = get_IDRiD_dataset(self.data_dir, DatasetVariant.TEST, self.img_size)
        super().setup(stage)


class EyePACSDataModule(FundusClassificationDatamodule):
    def setup(self, stage: str) -> None:
        if stage in ["fit", "validate"]:
            self.train = get_EyePACS_dataset(self.data_dir, DatasetVariant.TRAIN, self.img_size)
        if stage == "test":
            self.test = get_EyePACS_dataset(self.data_dir, DatasetVariant.TEST, self.img_size)
        super().setup(stage)

class AptosDataModule(FundusClassificationDatamodule):
    def setup(self, stage: str) -> None:
        if stage in ["fit", "validate"]:
            self.train = get_Aptos_dataset(self.data_dir, DatasetVariant.TRAIN, self.img_size)
        super().setup(stage)