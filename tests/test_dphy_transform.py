import numpy as np
import pytest

from core.cphy_transform import CPhyTransformer
from core.dphy_transform import DPhyTransformer
from core.models import Dataset
from core.network_transform import extract_dataset, transform_dataset
from core.sparameter_math import dphy_parameter_names, parse_dphy_parameter, se_parameter_names


def make_dphy_dataset(s, mapping=(1, 2, 3, 4)):
    return Dataset(
        "lane", "", np.arange(s.shape[0], dtype=float) + 1e9, s, 50,
        mapping, network_type="dphy", impedance_preset="DIFF nominal",
    )


def test_dphy_basis_is_orthonormal():
    matrix = DPhyTransformer.transform4()
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(4), atol=1e-14)


def test_sdd11_matches_standard_two_wire_expansion():
    rng = np.random.default_rng(44)
    s = rng.normal(size=(3, 4, 4)) + 1j * rng.normal(size=(3, 4, 4))
    modal = DPhyTransformer.transform(s)
    expected = (s[:, 0, 0] - s[:, 0, 1] - s[:, 1, 0] + s[:, 1, 1]) / 2
    np.testing.assert_allclose(modal[:, 0, 0], expected)


def test_diff_nominal_dataset_matches_standard_expansion_without_second_renormalization():
    rng = np.random.default_rng(45)
    s = (rng.normal(size=(3, 4, 4)) + 1j * rng.normal(size=(3, 4, 4))) * 0.01
    dataset = make_dphy_dataset(s)
    transform_dataset(dataset)
    np.testing.assert_allclose(dataset.dphy_modal, DPhyTransformer.transform(s))


def test_diff_custom_target_is_renormalized_from_intrinsic_nominal_references():
    rng = np.random.default_rng(46)
    s = (rng.normal(size=(3, 4, 4)) + 1j * rng.normal(size=(3, 4, 4))) * 0.01
    dataset = Dataset(
        "custom", "", np.arange(3, dtype=float) + 1e9, s, 50,
        network_type="dphy", impedance_preset="Custom", modal_reference_z0=(50, 50),
    )
    transform_dataset(dataset)
    expected = CPhyTransformer.renormalize(
        DPhyTransformer.transform(s), [100, 25, 100, 25], [50, 50, 50, 50]
    )
    np.testing.assert_allclose(dataset.dphy_modal, expected)


def test_dphy_catalog_contains_all_16_mixed_mode_parameters():
    names = dphy_parameter_names()
    assert len(names) == 16
    assert len(set(names)) == 16
    assert parse_dphy_parameter("Sdd21") == ("d", "d", 2, 1)
    assert parse_dphy_parameter("Scd12") == ("c", "d", 1, 2)
    for name in names:
        parse_dphy_parameter(name)


def test_dphy_dataset_exposes_mixed_mode_and_16_se_traces():
    s = np.zeros((2, 4, 4), complex)
    s[:, 2, 0] = 0.4
    dataset = make_dphy_dataset(s)
    transform_dataset(dataset)
    assert len(se_parameter_names(4)) == 16
    assert extract_dataset(dataset, "S31").shape == (2,)
    assert extract_dataset(dataset, "Sdd21").shape == (2,)
    with pytest.raises(ValueError, match="not valid"):
        extract_dataset(dataset, "S66")


def test_dphy_port_mapping_reorders_se_response_and_stimulus():
    s = np.zeros((1, 4, 4), complex)
    s[0, 1, 3] = 0.75
    dataset = make_dphy_dataset(s, mapping=(2, 1, 4, 3))
    transform_dataset(dataset)
    np.testing.assert_allclose(extract_dataset(dataset, "S13"), [0.75])


def test_invalid_dphy_mapping_is_rejected():
    dataset = make_dphy_dataset(np.zeros((1, 4, 4), complex), mapping=(1, 1, 3, 4))
    with pytest.raises(ValueError, match="1 through 4"):
        transform_dataset(dataset)
