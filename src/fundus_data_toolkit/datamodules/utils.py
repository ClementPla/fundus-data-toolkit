from typing import List

from nntools.dataset.utils.concat import concat_datasets_if_needed

from fundus_data_toolkit.datamodules.common import FundusDatamodule


def merge_existing_datamodules(
    datamodules: List[FundusDatamodule], separate_test_sets: bool = True
) -> FundusDatamodule:
    if len(datamodules) == 0:
        raise ValueError("No datamodules to merge.")

    if len(datamodules) == 1:
        return datamodules[0]

    assert all(
        isinstance(dm, FundusDatamodule) for dm in datamodules
    ), "All datamodules must be of type FundusDatamodule"

    img_size = set([dm.img_size for dm in datamodules])
    num_workers = set([dm.num_workers for dm in datamodules])
    batch_size = set([dm.batch_size for dm in datamodules])

    assert len(img_size) == 1, "All datamodules must have the same img_size"
    assert len(num_workers) == 1, "All datamodules must have the same num_workers"
    assert len(batch_size) == 1, "All datamodules must have the same batch_size"

    new_datamodule = FundusDatamodule(
        img_size=img_size.pop(), batch_size=batch_size.pop(), num_workers=num_workers.pop()
    )

    new_datamodule.train = concat_datasets_if_needed([dm.train for dm in datamodules if dm.train is not None])
    new_datamodule.val = concat_datasets_if_needed([dm.val for dm in datamodules if dm.val is not None])
    if separate_test_sets:
        new_datamodule.test = [dm.test for dm in datamodules if dm.test is not None]
    else:
        new_datamodule.test = concat_datasets_if_needed([dm.test for dm in datamodules if dm.test is not None])

    return new_datamodule
