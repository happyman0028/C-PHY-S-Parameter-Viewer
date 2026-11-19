from __future__ import annotations

from .cphy_transform import CPhyTransformer
from .dphy_transform import DPhyTransformer
from .models import Dataset
from .sparameter_math import is_dphy_parameter, is_se_parameter, parse_se_parameter


def transform_dataset(dataset: Dataset):
    if dataset.network_type == "dphy":
        return DPhyTransformer.transform_dataset(dataset)
    return CPhyTransformer.transform_dataset(dataset)


def extract_dataset(dataset: Dataset, parameter: str):
    if is_se_parameter(parameter):
        if dataset.s_se_common is None:
            transform_dataset(dataset)
        response_port, stimulus_port = parse_se_parameter(parameter)
        if response_port > dataset.nports or stimulus_port > dataset.nports:
            raise ValueError(f"{parameter} is not valid for a {dataset.nports}-port dataset.")
        return dataset.s_se_common[:, response_port - 1, stimulus_port - 1]
    if is_dphy_parameter(parameter):
        if dataset.network_type != "dphy":
            raise ValueError("DIFF parameters require an S4P DIFF dataset.")
        if dataset.dphy_modal is None:
            transform_dataset(dataset)
        return DPhyTransformer.extract(dataset.dphy_modal, parameter)
    if dataset.network_type != "cphy":
        raise ValueError("C-PHY parameters require an S6P C-PHY dataset.")
    from .sparameter_math import parse_parameter
    *_, r_index = parse_parameter(parameter)
    return CPhyTransformer.extract(dataset.modal[r_index], parameter)
