from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import Dataset
from .network_transform import normalize_se_dataset


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


def load_touchstone(
    path: str | Path,
    port_mapping: tuple[int, ...] | None = None,
    se_reference_z0: float | None = None,
    modal_reference_z0: tuple[float, ...] | None = None,
    impedance_preset: str | None = None,
    network_type: str | None = None,
) -> Dataset:
    """Load any Touchstone S-parameter file; S4P/S6P gain modal capabilities."""
    return _load_touchstone(
        path, None, network_type, port_mapping, se_reference_z0,
        modal_reference_z0, impedance_preset,
    )


def _load_touchstone(
    path: str | Path,
    expected_ports: int | None,
    network_type: str | None,
    port_mapping: tuple[int, ...] | None,
    se_reference_z0: float | None,
    modal_reference_z0: tuple[float, ...] | None,
    impedance_preset: str | None,
) -> Dataset:
    path = Path(path)
    if not path.exists():
        raise TouchstoneError(f"File does not exist: {path}")
    import re
    suffix_match = re.fullmatch(r"\.s(\d+)p", path.suffix, re.I)
    if not suffix_match:
        raise TouchstoneError("Expected a Touchstone .sNp file.")
    if expected_ports is not None and int(suffix_match.group(1)) != expected_ports:
        raise TouchstoneError(f"Expected a Touchstone .s{expected_ports}p file.")
    try:
        import skrf as rf
        network = rf.Network(str(path))
    except Exception as exc:
        raise TouchstoneError(f"Unable to read Touchstone file: {exc}") from exc
    if expected_ports is not None and network.nports != expected_ports:
        raise TouchstoneError(f"Expected a {expected_ports}-port network, received {network.nports} ports.")
    if int(suffix_match.group(1)) != network.nports:
        raise TouchstoneError(f"Filename declares {suffix_match.group(1)} ports, but the file contains {network.nports}.")
    network_type = network_type or ("dphy" if network.nports == 4 else "cphy" if network.nports == 6 else "se")
    port_mapping = port_mapping or tuple(range(1, network.nports + 1))
    impedance_preset = impedance_preset or ({"cphy": "C-PHY nominal", "dphy": "DIFF nominal"}.get(network_type, "SE only"))
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
    # Keep import responsive: only the mapped/common-Z0 SE matrix is prepared.
    # DIFF and C-PHY modal matrices are calculated on first use.
    normalize_se_dataset(dataset)
    return dataset
