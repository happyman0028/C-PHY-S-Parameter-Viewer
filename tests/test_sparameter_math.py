import numpy as np

from core.sparameter_math import DisplayFormat, format_trace


def test_log_magnitude_avoids_negative_infinity():
    result = format_trace(np.array([0j, 1 + 0j]), np.array([1e9, 2e9]), DisplayFormat.LOG_MAG.value)
    assert np.isfinite(result).all()
    assert result[1] == 0


def test_group_delay_uses_unwrapped_phase_and_nonuniform_frequency():
    frequency = np.array([1.0e9, 1.4e9, 2.1e9, 3.0e9, 4.2e9])
    delay = 73e-12
    values = np.exp(-1j * 2 * np.pi * frequency * delay)
    result = format_trace(values, frequency, DisplayFormat.GROUP_DELAY_PS.value)
    np.testing.assert_allclose(result, 73, atol=1e-9)


def test_group_delay_masks_transmission_zero():
    frequency = np.linspace(1e9, 2e9, 5)
    values = np.ones(5, complex)
    values[2] = 0
    result = format_trace(values, frequency, DisplayFormat.GROUP_DELAY_NS.value)
    assert np.isnan(result[2])
