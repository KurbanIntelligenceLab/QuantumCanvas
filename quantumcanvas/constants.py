"""Dataset-wide constants: file location, channel layout, label keys and units."""

DEFAULT_DATASET_PATH = "dataset_combined.npz"
ZENODO_DOI = "10.5281/zenodo.20631934"
ZENODO_URL = "https://zenodo.org/records/20631934/files/dataset_combined.npz?download=1"
DATASET_MD5 = "a35d349814ca9e12a8413289c015de49"

N_SAMPLES = 2850
IMAGE_SHAPE = (10, 32, 32)

# Index -> (name, description) for the ten image channels I_0 ... I_9.
CHANNELS = [
    ("orbital_population", "Orbital-weighted population stamp per atom"),
    ("angular_moment", "Net magnetic-moment magnitude per atom"),
    ("sp_shell_field", "Isotropic radial field x total s+p population"),
    ("df_shell_field", "Four-fold radial field x total d+f population"),
    ("dipole_field", "Radial ring x dipole magnitude |mu|"),
    ("charge_asymmetry_field", "Quadrupole field x |q_A - q_B|"),
    ("charge_magnitude", "Stamp per atom x |q|"),
    ("electron_population", "Stamp per atom x total electron population"),
    ("positive_charge", "Stamp at atoms with q > 0"),
    ("negative_charge", "Stamp at atoms with q < 0"),
]

# Label key -> unit for the targets reported in the paper's benchmark table.
# The three charge statistics are one quantity (q_B = -q_A, so all equal |q_A|),
# which is why these 19 keys make 17 distinct benchmark quantities.
BENCHMARK_TARGETS = {
    "e_g_ev": "eV",
    "e_homo_ev": "eV",
    "e_lumo_ev": "eV",
    "band_energy_ev": "eV",
    "total_energy_ev": "eV",
    "repulsive_energy_ev": "eV",
    "mermin_free_energy_ev": "eV",
    "i_ev": "eV",
    "a_ev": "eV",
    "chi_ev": "eV",
    "mu_ev": "eV",
    "eta_ev": "eV",
    "softness_evinv": "1/eV",
    "electrophilicity_ev": "eV",
    "dipole_mag_d": "D",
    "dipole_z_d": "D",
    "q_maxabs": "e",
    "q_absmean": "e",
    "q_std": "e",
}
REDUNDANT_CHARGE_TARGETS = ("q_maxabs", "q_absmean", "q_std")

ELEMENT_TO_Z = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
    'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
    'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
    'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
    'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
    'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
    'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98,
    'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103,
    'X': 0,
}
