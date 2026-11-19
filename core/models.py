from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np


@dataclass
class Dataset:
    name: str
    path: str
    frequency_hz: np.ndarray
    s_se: np.ndarray
    z0: float
    port_mapping: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    id: str = field(default_factory=lambda: uuid4().hex)
    modal: dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    source_z0: np.ndarray | None = field(default=None, repr=False)
    se_reference_z0: float | None = None
    modal_reference_z0: tuple[float, ...] | None = None
    impedance_preset: str = "C-PHY nominal"
    s_se_common: np.ndarray | None = field(default=None, repr=False)
    network_type: str = "cphy"
    dphy_modal: np.ndarray | None = field(default=None, repr=False)
    transform_revision: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if self.network_type not in {"cphy", "dphy"}:
            raise ValueError("Network type must be 'cphy' or 'dphy'.")
        if self.network_type == "dphy" and self.impedance_preset in {"C-PHY nominal", "D-PHY nominal"}:
            self.impedance_preset = "DIFF nominal"
        if len(self.port_mapping) != self.nports and self.port_mapping == (1, 2, 3, 4, 5, 6):
            self.port_mapping = tuple(range(1, self.nports + 1))
        if self.source_z0 is None:
            self.source_z0 = np.full((self.points, self.nports), complex(self.z0), dtype=complex)
        else:
            values = np.asarray(self.source_z0, dtype=complex)
            if values.ndim == 1:
                values = np.broadcast_to(values, (self.points, self.nports)).copy()
            self.source_z0 = values
        if self.se_reference_z0 is None:
            self.se_reference_z0 = float(self.z0)
        if self.modal_reference_z0 is None:
            value = float(self.se_reference_z0)
            if self.network_type == "dphy":
                self.modal_reference_z0 = (
                    (2.0 * value, value / 2.0)
                    if self.impedance_preset == "DIFF nominal"
                    else (value, value)
                )
            else:
                self.modal_reference_z0 = (
                    (2.0 * value, 1.5 * value, value / 3.0)
                    if self.impedance_preset == "C-PHY nominal"
                    else (value, value, value)
                )

    @property
    def nports(self) -> int:
        return int(np.asarray(self.s_se).shape[1])

    @property
    def file_type(self) -> str:
        return f"S{self.nports}P"

    @property
    def analysis_label(self) -> str:
        return "C-PHY" if self.network_type == "cphy" else "DIFF"

    @property
    def points(self) -> int:
        return int(self.frequency_hz.size)

    @property
    def frequency_range(self) -> tuple[float, float]:
        return float(self.frequency_hz[0]), float(self.frequency_hz[-1])

    @property
    def filename(self) -> str:
        return Path(self.path).name if self.path else self.name

    @property
    def source_z0_summary(self) -> str:
        values = np.asarray(self.source_z0).real
        low, high = float(np.min(values)), float(np.max(values))
        if np.allclose(values, low, rtol=1e-7, atol=1e-7):
            return f"{low:g} Ω"
        return f"variable {low:g}–{high:g} Ω"


@dataclass
class TraceConfig:
    dataset_id: str
    parameter: str
    color: str = "#4fc3f7"
    line_style: str = "Solid"
    visible: bool = True

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "parameter": self.parameter,
            "color": self.color,
            "line_style": self.line_style,
            "visible": self.visible,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TraceConfig":
        return cls(**data)


@dataclass
class MarkerConfig:
    name: str
    frequency_hz: float
    color: str
    visible: bool = True
    id: str = field(default_factory=lambda: uuid4().hex)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "frequency_hz": self.frequency_hz,
            "color": self.color,
            "visible": self.visible,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MarkerConfig":
        return cls(**data)


@dataclass
class PlotConfig:
    name: str
    display_format: str = "Log Magnitude (dB)"
    group_delay_unit: str = "ps"
    traces: list[TraceConfig] = field(default_factory=list)
    markers: list[MarkerConfig] = field(default_factory=list)
    frequency_range_hz: tuple[float, float] | None = None
    y_range: tuple[float, float] | None = None
    frequency_step_ghz: float | None = None
    y_step: float | None = None
    legend_visible: bool = True
    legend_font_size: int = 9
    title_font_size: int = 11
    axis_font_size: int = 9
    marker_info_expanded: bool = True
    marker_info_width: int = 350
    id: str = field(default_factory=lambda: uuid4().hex)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "display_format": self.display_format,
            "group_delay_unit": self.group_delay_unit,
            "traces": [trace.to_dict() for trace in self.traces],
            "markers": [marker.to_dict() for marker in self.markers],
            "frequency_range_hz": self.frequency_range_hz,
            "y_range": self.y_range,
            "frequency_step_ghz": self.frequency_step_ghz,
            "y_step": self.y_step,
            "legend_visible": self.legend_visible,
            "legend_font_size": self.legend_font_size,
            "title_font_size": self.title_font_size,
            "axis_font_size": self.axis_font_size,
            "marker_info_expanded": self.marker_info_expanded,
            "marker_info_width": self.marker_info_width,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlotConfig":
        payload = dict(data)
        payload["traces"] = [TraceConfig.from_dict(item) for item in payload.get("traces", [])]
        payload["markers"] = [MarkerConfig.from_dict(item) for item in payload.get("markers", [])]
        if payload.get("frequency_range_hz"):
            payload["frequency_range_hz"] = tuple(payload["frequency_range_hz"])
        if payload.get("y_range"):
            payload["y_range"] = tuple(payload["y_range"])
        return cls(**payload)
