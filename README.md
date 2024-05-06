![header](imgs/header.png)


# The Fundus Data Toolkit

This library provides a simple common interface to interact with the structures of different publicly available fundus databases.


## Datasets and DataModules

Internally, we separate the concepts of Datasets, Dataloaders and Datamodules.

- Datasets come from the [nntools library](https://github.com/ClementPla/NNTools/blob/main/src/nntools/dataset/abstract_image_dataset.py) but are in the end simply inherited from the PyTorch's dataset with a bunch of conveniance functions.
- [Dataloaders](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) are wrappers around datasets to create batches of samples.
- [Datamodule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) is a factory/container that create and keep the reference to the typical train/val/test splits (i.e the datasets). It also creates the dataloaders when asked. Finally, all preprocessing, data augmentation functions, caching etc... are configure inside the setup of the Datamodule. 

We preserve the train/val/test splits if they exist in the original databases. But most of the time, there are only two out three provided (train/val or train/test).  If train/val is provided, we consider val as test.

Eventually, if not proper validation set exists, we provide an option to split the train set in two. 
In practise, the datamodule creates the datasets in charge of each split. When used in training, the datamodules also returns the Dataloader associated to each dataset. 
In addition, we provide simple wrappers to merge datamodules or datasets.

## Configurations

The root folder must be configured at least once by calling:
```python
from fundus_data_toolkit.datamodules import Task, register_paths
paths = {'IDRID': 'path/to/IDRID', 'FGADR': 'path/to/FGADR'}
...
register_paths(paths, Task.SEGMENTATION)

paths = {'APTOS': 'path/to/APTOS', 'EYEPACS': 'path/to/EYEPACS'}
...
register_paths(paths, Task.CLASSIFICATION)

```

They can then be called directly in your code:

```python
from fundus_data_toolkit.datamodules.classification import IDRiDDataModule
from fundus_data_toolkit.datamodules import CLASSIF_PATHS, SEG_PATHS

# CLASSIF_PATHS.APTOS, CLASSIF_PATHS.EYEPACS, SEG_PATHS.IDRID, SEG_PATHS.FGADR

idrid_datamodule = IDRiDDataModule(CLASSIF_PATHS.IDRID, img_size=img_size, batch_size=8)
idrid_datamodule.setup_all()  # Create the train/val/test splits, either from existing splits or from the argments provided to the datamodule
```

Note that the ```register_paths``` function stores the paths in the user's setting and therefore does not need to be called more than once (if the data do not change).

## Installation

```bash
pip install .
```

or
```bash
pip install -e .
```