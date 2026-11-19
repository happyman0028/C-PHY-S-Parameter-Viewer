from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import Dataset
from .network_transform import transform_dataset


class TouchstoneError(ValueError):
    pass


def load_s6p(
    path: str | Path,
    port_mapping: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    se_reference_z0: float | None = None,
    modal_reference_z0: tuple[float, float, float] | None = None,
    impedance_preset: str = "C-PHY nominal",
) -> Dataset:
    return _load_touchstone(
        path, 6, "cphy", port_mapping, se_reference_z0,
        modal_reference_z0, impedance_preset,
    )


def load_s4p(
    path: str | Path,
    port_mapping: tuple[int, ...] = (1, 2, 3, 4),
    se_reference_z0: float | None = None,
    modal_reference_z0: tuple[float, float] | None = None,
    impedance_preset: str = "DIFF nominal",
) -> Dataset:
    return _load_touchstone(
        path, 4, "dphy", port_mapping, se_reference_z0,
        modal_reference_z0, impedance_preset,
    )


def _load_touchstone(
    path: str | Path,
    expected_ports: int,
    network_type: str,
    port_mapping: tuple[int, ...],
    se_reference_z0: float | None,
    modal_reference_z0: tuple[float, ...] | None,
    impedance_preset: str,
) -> Dataset:
    path = Path(path)
    if not path.exists():
        raise TouchstoneError(f"File does not exist: {path}")
    expected_suffix = f".s{expected_ports}p"
    if path.suffix.lower() != expected_suffix:
        raise TouchstoneError(f"Expected a Touchstone {expected_suffix} file.")
    try:
        import skrf as rf
        network = rf.Network(str(path))
    except Exception as exc:
        raise TouchstoneError(f"Unable to read Touchstone file: {exc}") from exc
    if network.nports != expected_ports:
        raise TouchstoneError(f"Expected a {expected_ports}-port network, received {network.nports} ports.")
    z0 = np.asarray(network.z0)
    if np.max(np.abs(z0.imag)) > 1e-9 or np.any(~np.isfinite(z0.real)) or np.any(z0.real <= 0):
        raise TouchstoneError("Reference impedances must be positive finite real values.")
    reference = float(z0.real.flat[0])
    frequency = np.asarray(network.f, dtype=float)
    s_values = np.asarray(network.s, dtype=complex)
    if frequency.size < 2 or not np.all(np.diff(frequency) > 0):
        raise TouchstoneError("Frequency points must be strictly increasing and contain at least two points.")
    dataset = Dataset(
        name=path.stem,
        path=str(path.resolve()),
        frequency_hz=frequency,
        s_se=s_values,
        z0=reference,
        port_mapping=tuple(int(value) for value in port_mapping),
        source_z0=z0,
        se_reference_z0=se_reference_z0 if se_reference_z0 is not None else reference,
        modal_reference_z0=modal_reference_z0,
        impedance_preset=impedance_preset,
        network_type=network_type,
    )
    transform_dataset(dataset)
    return dataset
