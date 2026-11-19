from __future__ import annotations

import re
from enum import Enum

import numpy as np

PARAMETER_RE = re.compile(r"^S([xyc])([xyc])([12])([12])_([123])R$", re.I)
SE_PARAMETER_RE = re.compile(r"^S([1-6])([1-6])$", re.I)
DPHY_PARAMETER_RE = re.compile(r"^S([dc])([dc])([12])([12])$", re.I)


class DisplayFormat(str, Enum):
    LOG_MAG = "Log Magnitude (dB)"
    LINEAR_MAG = "Linear Magnitude"
    PHASE = "Phase (degree)"
    UNWRAPPED_PHASE = "Unwrapped Phase (degree)"
    GROUP_DELAY_PS = "Group Delay (ps)"
    GROUP_DELAY_NS = "Group Delay (ns)"
    REAL = "Real"
    IMAGINARY = "Imaginary"


def parameter_names() -> list[str]:
    return [
        f"S{response}{stimulus}{response_side}{stimulus_side}_{r_index}R"
        for r_index in (1, 2, 3)
        for response in "xyc"
        for stimulus in "xyc"
        for response_side in (1, 2)
        for stimulus_side in (1, 2)
    ]


def se_parameter_names(nports: int = 6) -> list[str]:
    """Return all logical-DUT single-ended S-parameter names."""
    if nports not in (4, 6):
        raise ValueError("Single-ended catalog supports 4-port or 6-port datasets.")
    return [f"S{response}{stimulus}" for response in range(1, nports + 1) for stimulus in range(1, nports + 1)]


def parse_se_parameter(name: str) -> tuple[int, int]:
    match = SE_PARAMETER_RE.fullmatch(name.strip())
    if not match:
        raise ValueError(f"Invalid single-ended parameter name: {name}")
    return int(match.group(1)), int(match.group(2))


def is_se_parameter(name: str) -> bool:
    return SE_PARAMETER_RE.fullmatch(name.strip()) is not None


def dphy_parameter_names() -> list[str]:
    return [
        f"S{response}{stimulus}{response_side}{stimulus_side}"
        for response in "dc" for stimulus in "dc"
        for response_side in (1, 2) for stimulus_side in (1, 2)
    ]


def parse_dphy_parameter(name: str) -> tuple[str, str, int, int]:
    match = DPHY_PARAMETER_RE.fullmatch(name.strip())
    if not match:
        raise ValueError(f"Invalid DIFF mixed-mode parameter name: {name}")
    return match.group(1).lower(), match.group(2).lower(), int(match.group(3)), int(match.group(4))


def is_dphy_parameter(name: str) -> bool:
    return DPHY_PARAMETER_RE.fullmatch(name.strip()) is not None


def parse_parameter(name: str) -> tuple[str, str, int, int, int]:
    match = PARAMETER_RE.fullmatch(name.strip())
    if not match:
        raise ValueError(f"Invalid modal parameter name: {name}")
    return (
        match.group(1).lower(), match.group(2).lower(),
        int(match.group(3)), int(match.group(4)), int(match.group(5)),
    )


def format_trace(values: np.ndarray, frequency_hz: np.ndarray, display_format: str) -> np.ndarray:
    values = np.asarray(values, dtype=complex)
    frequency = np.asarray(frequency_hz, dtype=float)
    if values.shape != frequency.shape:
        raise ValueError("Trace and frequency arrays must have the same shape.")
    fmt = DisplayFormat(display_format)
    magnitude = np.abs(values)
    if fmt is DisplayFormat.LOG_MAG:
        return 20.0 * np.log10(np.maximum(magnitude, np.finfo(float).tiny))
    if fmt is DisplayFormat.LINEAR_MAG:
        return magnitude
    if fmt is DisplayFormat.PHASE:
        return np.rad2deg(np.angle(values))
    if fmt is DisplayFormat.UNWRAPPED_PHASE:
        return np.rad2deg(np.unwrap(np.angle(values)))
    if fmt in (DisplayFormat.GROUP_DELAY_PS, DisplayFormat.GROUP_DELAY_NS):
        phase = np.unwrap(np.angle(values))
        delay_seconds = -np.gradient(phase, frequency, edge_order=2) / (2 * np.pi)
        delay_seconds = delay_seconds.astype(float)
        delay_seconds[magnitude < 1e-10] = np.nan
        scale = 1e12 if fmt is DisplayFormat.GROUP_DELAY_PS else 1e9
        return delay_seconds * scale
    if fmt is DisplayFormat.REAL:
        return values.real
    return values.imag


def axis_label(display_format: str) -> str:
    return {
        DisplayFormat.LOG_MAG.value: "Magnitude (dB)",
        DisplayFormat.LINEAR_MAG.value: "Magnitude",
        DisplayFormat.PHASE.value: "Phase (degree)",
        DisplayFormat.UNWRAPPED_PHASE.value: "Unwrapped phase (degree)",
        DisplayFormat.GROUP_DELAY_PS.value: "Group delay (ps)",
        DisplayFormat.GROUP_DELAY_NS.value: "Group delay (ns)",
        DisplayFormat.REAL.value: "Real",
        DisplayFormat.IMAGINARY.value: "Imaginary",
    }[display_format]
