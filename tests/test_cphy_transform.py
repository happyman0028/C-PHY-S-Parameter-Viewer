import numpy as np
import pytest

from core.cphy_transform import CPhyTransformer
from core.models import Dataset
from core.sparameter_math import parameter_names, parse_parameter, parse_se_parameter, se_parameter_names


def test_all_r_bases_are_orthonormal():
    for r_index in (1, 2, 3):
        transform = CPhyTransformer.transform3(r_index)
        np.testing.assert_allclose(transform @ transform.T, np.eye(3), atol=1e-14)


def test_identity_is_invariant_under_transform():
    identity = np.broadcast_to(np.eye(6, dtype=complex), (4, 6, 6)).copy()
    for r_index in (1, 2, 3):
        np.testing.assert_allclose(CPhyTransformer.transform(identity, r_index), identity, atol=1e-14)


def test_ideal_thru_preserves_all_three_modes_and_has_no_conversion():
    s = np.zeros((3, 6, 6), dtype=complex)
    phase = np.exp(-1j * np.array([0.1, 0.7, 1.4]))
    for point, value in enumerate(phase):
        s[point, 3:6, 0:3] = np.eye(3) * value
        s[point, 0:3, 3:6] = np.eye(3) * value
    for r_index in (1, 2, 3):
        modal = CPhyTransformer.transform(s, r_index)
        np.testing.assert_allclose(
            np.diagonal(modal[:, 3:6, 0:3], axis1=1, axis2=2),
            np.repeat(phase[:, None], 3, axis=1),
        )
        off_diagonal = modal[:, 3:6, 0:3].copy()
        off_diagonal[:, np.arange(3), np.arange(3)] = 0
        np.testing.assert_allclose(off_diagonal, 0, atol=1e-14)


def test_raw1_xx11_matches_direct_expansion_and_keeps_complex_phase():
    rng = np.random.default_rng(123)
    s = rng.normal(size=(5, 6, 6)) + 1j * rng.normal(size=(5, 6, 6))
    modal = CPhyTransformer.transform(s, 1)
    expected = (s[:, 0, 0] - s[:, 0, 2] - s[:, 2, 0] + s[:, 2, 2]) / 2
    np.testing.assert_allclose(modal[:, 0, 0], expected)


def test_common_mode_subspace_is_identical_for_all_r_definitions():
    rng = np.random.default_rng(8)
    s = rng.normal(size=(2, 6, 6)) + 1j * rng.normal(size=(2, 6, 6))
    common_terms = []
    for r_index in (1, 2, 3):
        modal = CPhyTransformer.transform(s, r_index)
        common_terms.append(modal[:, [2, 2, 5, 5], [2, 5, 2, 5]])
    np.testing.assert_allclose(common_terms[0], common_terms[1])
    np.testing.assert_allclose(common_terms[1], common_terms[2])


def test_port_mapping_reorders_both_response_and_stimulus_axes():
    s = np.zeros((1, 6, 6), dtype=complex)
    s[0] = np.arange(36).reshape(6, 6)
    dataset = Dataset(
        "mapped", "", np.array([1e9]), s, 50, (3, 2, 1, 6, 5, 4),
        modal_reference_z0=(100, 75, 50 / 3), impedance_preset="C-PHY nominal",
    )
    modal = CPhyTransformer.transform_dataset(dataset)[1]
    reordered = s[:, [2, 1, 0, 5, 4, 3]][:, :, [2, 1, 0, 5, 4, 3]]
    np.testing.assert_allclose(modal, CPhyTransformer.transform(reordered, 1))


def test_parameter_catalog_has_108_unique_roundtrippable_names():
    names = parameter_names()
    assert len(names) == 108
    assert len(set(names)) == 108
    assert parse_parameter("Sxc12_1R") == ("x", "c", 1, 2, 1)
    assert parse_parameter("Sxy12_3R") == ("x", "y", 1, 2, 3)
    assert parse_parameter("Sxx21_2R") == ("x", "x", 2, 1, 2)
    for name in names:
        parse_parameter(name)


def test_invalid_mapping_is_rejected():
    dataset = Dataset("bad", "", np.array([1e9]), np.zeros((1, 6, 6), complex), 50, (1, 1, 2, 3, 4, 5))
    with pytest.raises(ValueError, match="each physical port"):
        CPhyTransformer.transform_dataset(dataset)


def test_cphy_nominal_matches_direct_expanded_modal_equations_without_second_renormalization():
    rng = np.random.default_rng(91)
    s = (rng.normal(size=(3, 6, 6)) + 1j * rng.normal(size=(3, 6, 6))) * 0.01
    dataset = Dataset(
        "nominal", "", np.array([1e9, 2e9, 3e9]), s, 50,
        modal_reference_z0=(100.0, 75.0, 50.0 / 3.0),
        impedance_preset="C-PHY nominal",
    )
    actual = CPhyTransformer.transform_dataset(dataset)[1]
    np.testing.assert_allclose(actual, CPhyTransformer.transform(s, 1))


def test_cphy_custom_target_is_renormalized_from_intrinsic_nominal_modal_references():
    rng = np.random.default_rng(92)
    s = (rng.normal(size=(3, 6, 6)) + 1j * rng.normal(size=(3, 6, 6))) * 0.01
    dataset = Dataset(
        "custom", "", np.array([1e9, 2e9, 3e9]), s, 50,
        modal_reference_z0=(50.0, 50.0, 50.0), impedance_preset="Custom",
    )
    actual = CPhyTransformer.transform_dataset(dataset)[1]
    direct_modal = CPhyTransformer.transform(s, 1)
    expected = CPhyTransformer.renormalize(
        direct_modal,
        [100, 75, 50 / 3, 100, 75, 50 / 3],
        [50, 50, 50, 50, 50, 50],
    )
    np.testing.assert_allclose(actual, expected)


def test_default_dataset_uses_cphy_nominal_reference_impedances():
    dataset = Dataset("default", "", np.array([1e9]), np.zeros((1, 6, 6), complex), 50)
    assert dataset.impedance_preset == "C-PHY nominal"
    np.testing.assert_allclose(dataset.modal_reference_z0, (100, 75, 50 / 3))


def test_single_ended_catalog_and_extraction_follow_logical_port_mapping():
    assert len(se_parameter_names()) == 36
    assert len(set(se_parameter_names())) == 36
    assert parse_se_parameter("S16") == (1, 6)
    s = np.zeros((1, 6, 6), complex)
    s[0, 2, 5] = 0.25
    dataset = Dataset("se", "", np.array([1e9]), s, 50, (3, 2, 1, 6, 5, 4))
    CPhyTransformer.transform_dataset(dataset)
    np.testing.assert_allclose(CPhyTransformer.extract_dataset(dataset, "S14"), [0.25])
