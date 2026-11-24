from __future__ import annotations

from .cphy_transform import CPhyTransformer
from .dphy_transform import DPhyTransformer
from .models import Dataset
from .sparameter_math import is_dphy_parameter, is_se_parameter, parse_se_parameter


def normalize_se_dataset(dataset: Dataset) -> np.ndarray:
    """Apply logical port mapping and common power-wave Z0 without modal work."""
    import numpy as np

    mapping = np.asarray(dataset.port_mapping, dtype=int) - 1
    if sorted(mapping.tolist()) != list(range(dataset.nports)):
        raise ValueError(f"Port mapping must contain each physical port 1 through {dataset.nports} exactly once.")
    reordered = dataset.s_se[:, mapping][:, :, mapping]
    source_z0 = np.asarray(dataset.source_z0, dtype=complex)[:, mapping]
    common_z0 = CPhyTransformer._positive_real(dataset.se_reference_z0, "Common SE Z0")
    dataset.s_se_common = CPhyTransformer.renormalize(reordered, source_z0, common_z0)
    return dataset.s_se_common


def transform_dataset(dataset: Dataset):
    if dataset.network_type == "dphy":
        return DPhyTransformer.transform_dataset(dataset)
    if dataset.network_type == "cphy":
        return CPhyTransformer.transform_dataset(dataset)
    dataset.modal.clear()
    dataset.dphy_modal = None
    return normalize_se_dataset(dataset)


def extract_dataset(dataset: Dataset, parameter: str):
    if is_se_parameter(parameter):
        if dataset.s_se_common is None:
            normalize_se_dataset(dataset)
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
    if r_index not in dataset.modal:
        transform_dataset(dataset)
    return CPhyTransformer.extract(dataset.modal[r_index], parameter)
