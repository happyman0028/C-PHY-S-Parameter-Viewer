from __future__ import annotations

import numpy as np

from .models import Dataset

MODES = ("x", "y", "c")
MODE_INDEX = {mode: index for index, mode in enumerate(MODES)}
R_DEFINITIONS = {
    1: np.array([[1, 0, -1], [-1, 2, -1], [1, 1, 1]], dtype=float),
    2: np.array([[-1, 1, 0], [-1, -1, 2], [1, 1, 1]], dtype=float),
    3: np.array([[0, -1, 1], [2, -1, -1], [1, 1, 1]], dtype=float),
}
NORMALIZERS = np.array([np.sqrt(2), np.sqrt(6), np.sqrt(3)])[:, None]


class CPhyTransformer:
    """Renormalize six-port data and convert its power waves into C-PHY modes."""

    @staticmethod
    def transform3(r_index: int) -> np.ndarray:
        if r_index not in R_DEFINITIONS:
            raise ValueError("R definition must be 1, 2, or 3.")
        matrix = R_DEFINITIONS[r_index] / NORMALIZERS
        if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-12):
            raise ValueError(f"{r_index}R basis is not orthonormal.")
        return matrix

    @classmethod
    def transform6(cls, r_index: int) -> np.ndarray:
        t3 = cls.transform3(r_index)
        t6 = np.zeros((6, 6), dtype=float)
        t6[:3, :3] = t3
        t6[3:, 3:] = t3
        return t6

    @classmethod
    def transform(cls, s_se: np.ndarray, r_index: int) -> np.ndarray:
        values = np.asarray(s_se, dtype=complex)
        if values.ndim != 3 or values.shape[1:] != (6, 6):
            raise ValueError("Expected complex S-parameter array shaped (frequency, 6, 6).")
        t6 = cls.transform6(r_index)
        return np.einsum("ab,fbc,dc->fad", t6, values, t6, optimize=True)

    @staticmethod
    def nominal_modal_z0(common_se_z0: float) -> tuple[float, float, float]:
        """Physical X/Y/CM references implied by the normalized mode basis."""
        value = CPhyTransformer._positive_real(common_se_z0, "Common SE Z0")
        return 2.0 * value, 1.5 * value, value / 3.0

    @classmethod
    def transform_dataset(cls, dataset: Dataset) -> dict[int, np.ndarray]:
        mapping = np.asarray(dataset.port_mapping, dtype=int) - 1
        if sorted(mapping.tolist()) != list(range(6)):
            raise ValueError("Port mapping must contain each physical port 1 through 6 exactly once.")
        reordered = dataset.s_se[:, mapping][:, :, mapping]
        source_z0 = np.asarray(dataset.source_z0, dtype=complex)[:, mapping]
        common_z0 = cls._positive_real(dataset.se_reference_z0, "Common SE Z0")
        modal_z0 = tuple(cls._positive_real(value, label) for value, label in zip(
            dataset.modal_reference_z0 or (), ("X-mode Z0", "Y-mode Z0", "CM-mode Z0")
        ))
        if len(modal_z0) != 3:
            raise ValueError("Modal reference impedance must contain X, Y, and CM values.")

        dataset.s_se_common = cls.renormalize(reordered, source_z0, common_z0)
        target = np.asarray((*modal_z0, *modal_z0), dtype=float)
        nominal = cls.nominal_modal_z0(common_z0)
        source_modal_z0 = np.asarray((*nominal, *nominal), dtype=float)
        dataset.modal = {}
        for r_index in (1, 2, 3):
            equal_power_modal = cls.transform(dataset.s_se_common, r_index)
            # The orthonormal wave transform already represents the physical
            # X/Y/CM definitions at 2×SE, 1.5×SE, and SE÷3. Treating these
            # waves as three 50-ohm modes and renormalizing again double-counts
            # the modal scaling. Only a genuinely custom target needs a
            # post-transform renormalization.
            dataset.modal[r_index] = cls.renormalize(equal_power_modal, source_modal_z0, target)
        dataset.transform_revision += 1
        return dataset.modal

    @staticmethod
    def _positive_real(value: float | None, label: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be a positive finite real number.") from exc
        if not np.isfinite(result) or result <= 0:
            raise ValueError(f"{label} must be a positive finite real number.")
        return result

    @staticmethod
    def renormalize(s_values: np.ndarray, z_old, z_new) -> np.ndarray:
        """Power-wave renormalization supporting per-frequency, per-port source Z0."""
        import skrf as rf

        old = np.asarray(z_old, dtype=complex)
        if np.max(np.abs(old.imag)) > 1e-9 or np.any(~np.isfinite(old.real)) or np.any(old.real <= 0):
            raise ValueError("Source reference impedances must be positive finite real values.")
        if np.allclose(old.real, np.asarray(z_new, dtype=float), rtol=1e-12, atol=1e-12):
            return np.asarray(s_values, complex).copy()
        return np.asarray(rf.renormalize_s(np.asarray(s_values, complex), old.real, z_new, s_def="power"))

    @staticmethod
    def modal_index(mode: str, side: int) -> int:
        mode = mode.lower()
        if mode not in MODE_INDEX or side not in (1, 2):
            raise ValueError("Mode must be x/y/c and side must be 1/2.")
        return (side - 1) * 3 + MODE_INDEX[mode]

    @classmethod
    def extract(cls, modal: np.ndarray, parameter: str) -> np.ndarray:
        from .sparameter_math import parse_parameter

        response_mode, stimulus_mode, response_side, stimulus_side, _ = parse_parameter(parameter)
        row = cls.modal_index(response_mode, response_side)
        column = cls.modal_index(stimulus_mode, stimulus_side)
        return modal[:, row, column]

    @classmethod
    def extract_dataset(cls, dataset: Dataset, parameter: str) -> np.ndarray:
        """Extract either a logical-port SE trace or a C-PHY modal trace."""
        from .sparameter_math import is_se_parameter, parse_parameter, parse_se_parameter

        if is_se_parameter(parameter):
            if dataset.s_se_common is None:
                cls.transform_dataset(dataset)
            response_port, stimulus_port = parse_se_parameter(parameter)
            return dataset.s_se_common[:, response_port - 1, stimulus_port - 1]
        *_, r_index = parse_parameter(parameter)
        return cls.extract(dataset.modal[r_index], parameter)
