"""Generate a deterministic six-port C-PHY-like demo Touchstone file."""
from pathlib import Path

import numpy as np
import skrf as rf


def build_demo() -> rf.Network:
    frequency = rf.Frequency.from_f(np.linspace(0.1, 12.0, 401), unit="ghz")
    f = frequency.f
    s = np.zeros((f.size, 6, 6), dtype=complex)
    for port in range(6):
        reflection = 0.04 + 0.012 * np.sin(f / 1.1e9 + port * 0.5)
        s[:, port, port] = reflection * np.exp(-1j * 2 * np.pi * f * (45e-12 + port * 2e-12))
    transmission = 0.88 * np.exp(-f / 45e9) * np.exp(-1j * 2 * np.pi * f * 310e-12)
    for wire in range(3):
        s[:, wire + 3, wire] = transmission
        s[:, wire, wire + 3] = transmission
    coupling = 0.014 * np.exp(-f / 30e9) * np.exp(-1j * 2 * np.pi * f * 260e-12)
    for response in range(6):
        for stimulus in range(6):
            if response != stimulus and np.all(s[:, response, stimulus] == 0):
                s[:, response, stimulus] = coupling * (1 + 0.08 * (response - stimulus))
    return rf.Network(frequency=frequency, s=s, z0=50, name="cphy_demo_6port")


if __name__ == "__main__":
    output = Path(__file__).resolve().parent
    build_demo().write_touchstone(dir=output, form="ri")
    print(output / "cphy_demo_6port.s6p")
