import numpy as np
import pytest

LABEL_KEYS = [
    'e_g_ev', 'e_homo_ev', 'e_lumo_ev', 'total_energy_ev', 'band_energy_ev', 'repulsive_energy_ev',
    'mermin_free_energy_ev', 'i_ev', 'a_ev', 'chi_ev', 'mu_ev', 'eta_ev', 'softness_evinv',
    'electrophilicity_ev', 'dipole_mag_d', 'dipole_z_d', 'q_maxabs', 'q_absmean', 'q_std', 'distance_ang',
]


@pytest.fixture(scope="session")
def synthetic_npz(tmp_path_factory):
    """A small file with the same keys, shapes and dtypes as dataset_combined.npz."""
    rng = np.random.default_rng(0)
    n = 24
    symbols = ['H', 'C', 'N', 'O', 'Fe', 'Ag', 'Al', 'Si']
    elements = np.array([[symbols[i % 8], symbols[(3 * i + 1) % 8]] for i in range(n)], dtype=object)
    geometries = np.zeros((n, 2, 4))
    geometries[:, 1, 0] = rng.uniform(1.5, 3.5, n)
    geometries[:, :, 3] = rng.uniform(1, 10, (n, 2))
    labels = [{k: float(rng.normal()) for k in LABEL_KEYS} | {'total_charge': 0.0} for _ in range(n)]
    for lab in labels:  # neutral pairs: the three charge statistics coincide
        lab['q_absmean'] = lab['q_std'] = lab['q_maxabs'] = abs(lab['q_maxabs'])
    labels[0]['dipole_mag_d'] = float('nan')  # missing values must be skipped
    path = tmp_path_factory.mktemp("data") / "synthetic.npz"
    np.savez(
        path,
        images=rng.random((n, 10, 32, 32)).astype(np.float32),
        geometries=geometries,
        elements=elements,
        labels=np.array(labels, dtype=object),
        metadata=np.array([{} for _ in range(n)], dtype=object),
        pair_names=np.array([f"{a}_{b}" for a, b in elements], dtype=object),
    )
    return str(path)
