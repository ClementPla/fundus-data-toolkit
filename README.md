![header](imgs/header.png)


# The Fundus Data Toolkit

This library provides a simple common interface to interact with the structures of different publicly available fundus databases.


## Datasets and DataModules

Internally, we separate the concepts of Datasets, Datadoaders and Datamodules.

- Datasets come from the [nntools library](https://github.com/ClementPla/NNTools/blob/main/src/nntools/dataset/abstract_image_dataset.py) but are in the end simply inherited from the PyTorch's dataset with a bunch of conveniance functions.
- [Dataloaders](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) are wrappers around datasets to create batches of samples.
- [Datamodule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) is a convenience container used to create and store the typical train/val/test splits. 

When the train/val/test splits already exist in the original databases, we maintain them. Most of the time, there is only two out three provided (train/val or train/test).  If train/val is provided, we consider val as test.

Eventually, if not proper validation set exists, we provide an option to split the train set in two. In practise, the datamodule creates the datasets in charge of each split. When used in training, the datamodules also returns the Dataloader associated to each dataset. In addition, we provide simple wrappers to merge datamodules.

## Configurations

The root folder(s) must be indicated in the configuration files (which is also the list of all datasets maintened in this library so far). 
They can then be called directly in your code:

```python
from fundusData.datamodules.classification import IDRiDDataModule
from fundusData.datamodules import CLASSIF_PATHS, SEG_PATHS
idrid_datamodule = IDRiDDataModule(CLASSIF_PATHS.IDRID, img_size=img_size, batch_size=8).setup_all()
```

## Installation

```bash
pip install .
```

or
```bash
pip install -e .
```