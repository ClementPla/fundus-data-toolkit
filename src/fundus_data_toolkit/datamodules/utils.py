from copy import deepcopy
from typing import List

from fundus_data_toolkit.datamodules.common import FundusDatamodule, MergedDatamodule


def merge_existing_datamodules(
    datamodules: List[FundusDatamodule], separate_test_sets: bool = True
) -> FundusDatamodule:
    new_datamodule = MergedDatamodule(*datamodules, separate_test_sets=separate_test_sets)

    return new_datamodule
