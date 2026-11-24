from __future__ import annotations

import numpy as np

from .cphy_transform import CPhyTransformer
from .models import Dataset


class DPhyTransformer:
    """Convert a four-port P+/P- network to DIFF differential/common modes."""

    @staticmethod
    def transform4() -> np.ndarray:
        scale = np.sqrt(2.0)
        return np.array([
            [1, -1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, -1],
            [0, 0, 1, 1],
        ], dtype=float) / scale

    @classmethod
    def transform(cls, s_se: np.ndarray) -> np.ndarray:
        values = np.asarray(s_se, dtype=complex)
        if values.ndim != 3 or values.shape[1:] != (4, 4):
            raise ValueError("Expected complex S-parameter array shaped (frequency, 4, 4).")
        matrix = cls.transform4()
        return np.einsum("ab,fbc,dc->fad", matrix, values, matrix, optimize=True)

    @classmethod
    def transform_dataset(cls, dataset: Dataset) -> np.ndarray:
        from .network_transform import normalize_se_dataset

        if dataset.nports != 4:
            raise ValueError("DIFF mixed-mode conversion requires a 4-port dataset.")
        common_se = CPhyTransformer._positive_real(dataset.se_reference_z0, "Common SE Z0")
        target_values = dataset.modal_reference_z0 or ()
        if len(target_values) != 2:
            raise ValueError("DIFF reference impedance must contain differential and common-mode values.")
        differential_z0 = CPhyTransformer._positive_real(target_values[0], "Differential Z0")
        common_z0 = CPhyTransformer._positive_real(target_values[1], "Common-mode Z0")
        if dataset.s_se_common is None:
            normalize_se_dataset(dataset)
        equal_power_modal = cls.transform(dataset.s_se_common)
        source_modal_z0 = [2.0 * common_se, common_se / 2.0] * 2
        dataset.dphy_modal = CPhyTransformer.renormalize(
            equal_power_modal, source_modal_z0,
            [differential_z0, common_z0, differential_z0, common_z0],
        )
        dataset.transform_revision += 1
        return dataset.dphy_modal

    @staticmethod
    def extract(modal: np.ndarray, parameter: str) -> np.ndarray:
        from .sparameter_math import parse_dphy_parameter

        response_mode, stimulus_mode, response_side, stimulus_side = parse_dphy_parameter(parameter)
        mode_index = {"d": 0, "c": 1}
        row = (response_side - 1) * 2 + mode_index[response_mode]
        column = (stimulus_side - 1) * 2 + mode_index[stimulus_mode]
        return modal[:, row, column]
