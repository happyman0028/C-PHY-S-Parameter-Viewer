from pathlib import Path

import numpy as np
import pytest
import skrf as rf

from core.network_transform import extract_dataset
from core.sparameter_math import se_parameter_names
from core.touchstone_loader import TouchstoneError, load_s4p, load_s6p, load_touchstone


def write_network(path: Path, z0=50.0, nports=6) -> Path:
    frequency = rf.Frequency.from_f(np.linspace(1, 3, 5), unit="ghz")
    s = np.zeros((5, nports, nports), dtype=complex)
    network = rf.Network(frequency=frequency, s=s, z0=z0, name=path.stem)
    network.write_touchstone(dir=path.parent, write_z0=not np.all(np.asarray(z0) == np.asarray(z0).flat[0]))
    return path


def test_load_valid_s6p(tmp_path):
    path = write_network(tmp_path / "valid.s6p")
    dataset = load_s6p(path)
    assert dataset.points == 5
    assert dataset.z0 == 50
    assert dataset.impedance_preset == "C-PHY nominal"
    np.testing.assert_allclose(dataset.modal_reference_z0, (100, 75, 50 / 3))
    assert dataset.modal == {}
    extract_dataset(dataset, "Sxx11_1R")
    assert set(dataset.modal) == {1, 2, 3}
    assert dataset.modal[1].shape == (5, 6, 6)


def test_unequal_reference_impedance_is_renormalized_to_common_se_z0(tmp_path):
    z0 = np.tile(np.array([50, 50, 50, 75, 75, 75]), (5, 1))
    path = write_network(tmp_path / "unequal.s6p", z0=z0)
    dataset = load_s6p(path)
    np.testing.assert_allclose(dataset.source_z0.real, z0)
    assert dataset.se_reference_z0 == 50
    assert dataset.source_z0_summary == "variable 50–75 Ω"
    assert dataset.s_se_common.shape == (5, 6, 6)
    assert dataset.modal == {}


def test_load_valid_s4p_as_dphy_with_nominal_impedances(tmp_path):
    path = write_network(tmp_path / "lane0.s4p", nports=4)
    dataset = load_s4p(path)
    assert dataset.nports == 4
    assert dataset.file_type == "S4P"
    assert dataset.analysis_label == "DIFF/SE"
    assert dataset.port_mapping == (1, 2, 3, 4)
    assert dataset.impedance_preset == "DIFF nominal"
    np.testing.assert_allclose(dataset.modal_reference_z0, (100, 25))
    assert dataset.dphy_modal is None
    assert extract_dataset(dataset, "Sdd11").shape == (5,)
    assert dataset.dphy_modal.shape == (5, 4, 4)
    assert dataset.s_se_common.shape == (5, 4, 4)


def test_s4p_and_s6p_loaders_reject_the_other_file_type(tmp_path):
    s4p = write_network(tmp_path / "lane.s4p", nports=4)
    s6p = write_network(tmp_path / "trio.s6p", nports=6)
    with pytest.raises(TouchstoneError, match=".s6p"):
        load_s6p(s4p)
    with pytest.raises(TouchstoneError, match=".s4p"):
        load_s4p(s6p)


@pytest.mark.parametrize("nports", [1, 3])
def test_generic_loader_supports_se_only_touchstone(nports, tmp_path):
    path = write_network(tmp_path / f"generic.s{nports}p", nports=nports)
    dataset = load_touchstone(path)
    assert dataset.nports == nports
    assert dataset.file_type == f"S{nports}P"
    assert dataset.network_type == "se"
    assert dataset.analysis_label == "SE"
    assert dataset.modal_reference_z0 == ()
    assert len(se_parameter_names(nports)) == nports ** 2
    assert extract_dataset(dataset, se_parameter_names(nports)[-1]).shape == (5,)


def test_se_access_does_not_trigger_modal_transform(tmp_path):
    dataset = load_touchstone(write_network(tmp_path / "lazy.s6p"))
    assert dataset.modal == {}
    extract_dataset(dataset, "S11")
    assert dataset.modal == {}
