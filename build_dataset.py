#!/usr/bin/env python3
"""
Complete Dataset Builder for Two-Body Quantum Systems

This script does EVERYTHING in one go:
1. Parses detailed.out files (orbital populations)
2. Parses geo_end.xyz files (3D coordinates)
3. Creates 10-channel image tensors [10, 32, 32]
4. Integrates all 48 labels from CSV files
5. Saves .npz files with images + geometry + labels

Usage:
    python build_dataset.py

Output:
    processed_images/*.npz - Complete data files
    analysis/ - Labels and statistics
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class OrbitalData:
    atom_idx: int
    shell: int
    l: int
    m: int
    population: float
    label: str


@dataclass
class AtomData:
    atom_idx: int
    charge: float
    total_population: float
    orbitals: List[OrbitalData]


@dataclass
class AtomGeometry:
    element: str
    x: float
    y: float
    z: float
    population: float


@dataclass
class MoleculeData:
    total_charge: float
    atoms: List[AtomData]
    fermi_level: float
    total_energy: float
    dipole_moment: Tuple[float, float, float]
    pair_name: str
    geometry: Optional[List[AtomGeometry]] = None
    bond_length: Optional[float] = None


# ============================================================================
# PARSERS
# ============================================================================

class GeometryParser:
    @staticmethod
    def parse_xyz(filepath: Path) -> Optional[Tuple[List[AtomGeometry], float]]:
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            num_atoms = int(lines[0].strip())
            geometry = []
            
            for i in range(2, 2 + num_atoms):
                parts = lines[i].split()
                atom = AtomGeometry(
                    element=parts[0],
                    x=float(parts[1]),
                    y=float(parts[2]),
                    z=float(parts[3]),
                    population=float(parts[4]) if len(parts) > 4 else 0.0
                )
                geometry.append(atom)
            
            if len(geometry) == 2:
                dx = geometry[1].x - geometry[0].x
                dy = geometry[1].y - geometry[0].y
                dz = geometry[1].z - geometry[0].z
                bond_length = np.sqrt(dx**2 + dy**2 + dz**2)
            else:
                bond_length = None
            
            return geometry, bond_length
        except:
            return None, None


class DetailedOutParser:
    @staticmethod
    def parse_file(filepath: Path) -> Optional[MoleculeData]:
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Extract data
            total_charge_match = re.search(r'Total charge:\s+([-+]?\d+\.\d+)', content)
            total_charge = float(total_charge_match.group(1)) if total_charge_match else 0.0
            
            charge_section = re.search(r'Atomic gross charges \(e\)\n Atom\s+Charge\n((?:\s+\d+\s+[-+]?\d+\.\d+\n)+)', content)
            atom_charges = {}
            if charge_section:
                for line in charge_section.group(1).strip().split('\n'):
                    parts = line.split()
                    atom_charges[int(parts[0])] = float(parts[1])
            
            pop_section = re.search(r'Atom populations \(up\)\n Atom\s+Population\n((?:\s+\d+\s+\d+\.\d+\n)+)', content)
            atom_pops = {}
            if pop_section:
                for line in pop_section.group(1).strip().split('\n'):
                    parts = line.split()
                    atom_pops[int(parts[0])] = float(parts[1])
            
            orbital_section = re.search(
                r'Orbital populations \(up\)\n Atom Sh\.\s+l\s+m\s+Population\s+Label\n((?:\s+\d+\s+\d+\s+\d+\s+[-+]?\d+\s+\d+\.\d+\s+\S+\n)+)',
                content
            )
            
            atoms_data = {}
            if orbital_section:
                for line in orbital_section.group(1).strip().split('\n'):
                    parts = line.split()
                    atom_idx = int(parts[0])
                    orbital = OrbitalData(atom_idx, int(parts[1]), int(parts[2]), 
                                        int(parts[3]), float(parts[4]), parts[5])
                    
                    if atom_idx not in atoms_data:
                        atoms_data[atom_idx] = []
                    atoms_data[atom_idx].append(orbital)
            
            atoms = []
            for atom_idx in sorted(atoms_data.keys()):
                atoms.append(AtomData(
                    atom_idx=atom_idx,
                    charge=atom_charges.get(atom_idx, 0.0),
                    total_population=atom_pops.get(atom_idx, 0.0),
                    orbitals=atoms_data[atom_idx]
                ))
            
            fermi_match = re.search(r'Fermi level:\s+([-+]?\d+\.\d+)\s+H', content)
            fermi_level = float(fermi_match.group(1)) if fermi_match else 0.0
            
            energy_match = re.search(r'Total energy:\s+([-+]?\d+\.\d+)\s+H', content)
            total_energy = float(energy_match.group(1)) if energy_match else 0.0
            
            dipole_match = re.search(r'Dipole moment:\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+au', content)
            dipole_moment = (0.0, 0.0, 0.0)
            if dipole_match:
                dipole_moment = (float(dipole_match.group(1)), 
                               float(dipole_match.group(2)), 
                               float(dipole_match.group(3)))
            
            pair_name = filepath.parent.name
            
            xyz_file = filepath.parent / "geo_end.xyz"
            geometry, bond_length = None, None
            if xyz_file.exists():
                geometry, bond_length = GeometryParser.parse_xyz(xyz_file)
            
            return MoleculeData(
                total_charge=total_charge,
                atoms=atoms,
                fermi_level=fermi_level,
                total_energy=total_energy,
                dipole_moment=dipole_moment,
                pair_name=pair_name,
                geometry=geometry,
                bond_length=bond_length
            )
        except:
            return None


# ============================================================================
# IMAGE ENCODER
# ============================================================================

class ImageEncoder:
    def __init__(self, image_size: int = 32):
        self.image_size = image_size
    
    def encode(self, mol_data: MoleculeData) -> np.ndarray:
        H, W = self.image_size, self.image_size
        image = np.zeros((10, H, W), dtype=np.float32)
        
        features = self._extract_features(mol_data)
        
        image[0:2] = self._generate_omap(mol_data, features)
        image[2:4] = self._generate_rip_gaf(mol_data, features)
        image[4:6] = self._generate_rip_mtf(mol_data, features)
        image[6:8] = self._generate_com(mol_data, features)
        image[8:10] = self._generate_q_image(mol_data, features)
        
        return image
    
    def _extract_features(self, mol_data):
        features = {'s_pop': [], 'p_pop': [], 'd_pop': [], 'f_pop': [], 
                   'total_pop': [], 'charge': [], 'm_moments': []}
        
        for atom in mol_data.atoms:
            features['s_pop'].append(sum(o.population for o in atom.orbitals if o.l == 0))
            features['p_pop'].append(sum(o.population for o in atom.orbitals if o.l == 1))
            features['d_pop'].append(sum(o.population for o in atom.orbitals if o.l == 2))
            features['f_pop'].append(sum(o.population for o in atom.orbitals if o.l == 3))
            features['total_pop'].append(atom.total_population)
            features['charge'].append(atom.charge)
            features['m_moments'].append(sum(abs(o.m) * o.population for o in atom.orbitals))
        
        return features
    
    def _generate_omap(self, mol, features):
        H, W = self.image_size, self.image_size
        omap = np.zeros((2, H, W), dtype=np.float32)
        n = len(mol.atoms)
        for idx, atom in enumerate(mol.atoms):
            x = W // 2 + (idx - n/2 + 0.5) * (W // (n + 1))
            y = H // 2
            for orb in atom.orbitals:
                self._add_gaussian(omap[0], x, y, orb.population * (orb.l + 1) / 3.0, 3.0)
            m_weighted = sum(orb.m * orb.population for orb in atom.orbitals)
            self._add_gaussian(omap[1], x, y, abs(m_weighted), 3.0)
        return omap
    
    def _generate_rip_gaf(self, mol, features):
        H, W = self.image_size, self.image_size
        rip = np.zeros((2, H, W), dtype=np.float32)
        y_grid, x_grid = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
        r = np.sqrt(x_grid**2 + y_grid**2)
        rip[0] = np.exp(-r**2 / 0.3) * (sum(features['s_pop']) + sum(features['p_pop']))
        rip[1] = np.exp(-r**2 / 0.5) * (sum(features['d_pop']) + sum(features['f_pop'])) * (1 + np.cos(4 * np.arctan2(y_grid, x_grid)))
        return rip
    
    def _generate_rip_mtf(self, mol, features):
        H, W = self.image_size, self.image_size
        rip = np.zeros((2, H, W), dtype=np.float32)
        y_grid, x_grid = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
        r = np.sqrt(x_grid**2 + y_grid**2) + 1e-6
        rip[0] = np.linalg.norm(mol.dipole_moment) * r * np.exp(-r**2 / 0.4)
        charge_diff = abs(features['charge'][0] - features['charge'][-1]) if len(features['charge']) >= 2 else 0
        rip[1] = charge_diff * (3 * np.cos(np.arctan2(y_grid, x_grid))**2 - 1) * np.exp(-r**2 / 0.4)
        return rip
    
    def _generate_com(self, mol, features):
        H, W = self.image_size, self.image_size
        com = np.zeros((2, H, W), dtype=np.float32)
        n = len(mol.atoms)
        for idx, (charge, pop) in enumerate(zip(features['charge'], features['total_pop'])):
            x = W // 2 + (idx - n/2 + 0.5) * (W // (n + 1))
            y = H // 2
            self._add_gaussian(com[0], x, y, abs(charge) * 10, 4.0)
            self._add_gaussian(com[1], x, y, pop, 4.0)
        return com
    
    def _generate_q_image(self, mol, features):
        H, W = self.image_size, self.image_size
        q = np.zeros((2, H, W), dtype=np.float32)
        n = len(mol.atoms)
        for idx, charge in enumerate(features['charge']):
            x = W // 2 + (idx - n/2 + 0.5) * (W // (n + 1))
            y = H // 2
            if charge > 0:
                self._add_gaussian(q[0], x, y, charge * 10, 3.5)
            else:
                self._add_gaussian(q[1], x, y, abs(charge) * 10, 3.5)
        return q
    
    def _add_gaussian(self, channel, x, y, amp, sigma):
        H, W = channel.shape
        y_grid, x_grid = np.ogrid[:H, :W]
        gaussian = amp * np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
        channel += gaussian


# ============================================================================
# LABEL LOADER
# ============================================================================

def load_labels(raw_data_dir):
    """Load and merge all CSV labels"""
    # Load combined CSV
    df = pd.read_csv(raw_data_dir / "dftb_ptbp_combined.csv")
    df['pair_name'] = df['run_dir'].apply(lambda x: Path(x).name)
    
    # Load bond distances
    df_bonds = pd.read_csv(raw_data_dir / "bond_distances_all.csv")
    df_bonds['pair_name'] = df_bonds['run_dir'].apply(lambda x: Path(x).name)
    
    # Merge geometry columns
    geom_cols = ['pair_name', 'natoms', 'species_1', 'x1', 'y1', 'z1', 
                 'species_2', 'x2', 'y2', 'z2', 'pair', 'pair_sorted', 'distance_Ang']
    df_bonds_subset = df_bonds[geom_cols].copy()
    
    df = df.merge(df_bonds_subset, on='pair_name', how='left', suffixes=('', '_bonds'))
    
    # Replace empty columns with bond data
    for col in ['distance_Ang', 'natoms', 'species_1', 'species_2', 
                'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'pair', 'pair_sorted']:
        if f'{col}_bonds' in df.columns:
            df[col] = df[f'{col}_bonds']
            df = df.drop(columns=[f'{col}_bonds'])
    
    df = df.drop(columns=['run_dir'])
    df = df.set_index('pair_name')
    
    return df


# ============================================================================
# MAIN PROCESSOR
# ============================================================================

def process_dataset(raw_data_dir, output_file="dataset_combined.npz"):
    """Process all samples and save to single file"""
    raw_data_dir = Path(raw_data_dir)
    
    print("=" * 70)
    print("TWO-BODY QUANTUM SYSTEM DATASET BUILDER")
    print("=" * 70)
    
    # Load labels
    print("\n[1/4] Loading labels from CSVs...")
    labels_df = load_labels(raw_data_dir)
    print(f"✓ Loaded {len(labels_df)} samples with {len(labels_df.columns)} labels")
    
    # Find all detailed.out files
    print("\n[2/4] Finding samples...")
    detailed_files = list(raw_data_dir.glob("*/detailed.out"))
    print(f"✓ Found {len(detailed_files)} samples")
    
    # Process each sample - collect in memory
    print("\n[3/4] Processing samples (images + geometry + labels)...")
    encoder = ImageEncoder()
    
    images = []
    geometries = []
    elements_list = []
    labels_list = []
    metadata_list = []
    pair_names = []
    
    for detailed_file in tqdm(detailed_files):
        mol_data = DetailedOutParser.parse_file(detailed_file)
        if mol_data is None:
            continue
        
        # Create image
        image = encoder.encode(mol_data)
        images.append(image)
        
        # Metadata
        metadata = {
            'pair_name': mol_data.pair_name,
            'total_charge': mol_data.total_charge,
            'fermi_level': mol_data.fermi_level,
            'total_energy': mol_data.total_energy,
            'dipole_moment': mol_data.dipole_moment,
            'num_atoms': len(mol_data.atoms),
            'bond_length': mol_data.bond_length,
        }
        metadata_list.append(metadata)
        pair_names.append(mol_data.pair_name)
        
        # Add geometry
        if mol_data.geometry:
            geometry_array = np.array([[a.x, a.y, a.z, a.population] for a in mol_data.geometry])
            geometries.append(geometry_array)
            elements_list.append([a.element for a in mol_data.geometry])
        else:
            geometries.append(np.zeros((2, 4)))
            elements_list.append(['X', 'X'])
        
        # Add labels (normalize names to lowercase, EXCLUDE geometry columns)
        if mol_data.pair_name in labels_df.index:
            labels = labels_df.loc[mol_data.pair_name].to_dict()
            
            # Exclude geometry columns (already in geometry array)
            exclude_keys = {'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'species_1', 'species_2',
                           'natoms', 'pair', 'pair_sorted'}
            
            labels_clean = {}
            for key, value in labels.items():
                key_normalized = key.lower()
                
                if key_normalized in exclude_keys:
                    continue
                
                try:
                    if pd.isna(value):
                        labels_clean[key_normalized] = None
                    elif isinstance(value, (int, float, np.number)):
                        labels_clean[key_normalized] = float(value) if not np.isinf(value) else None
                    else:
                        labels_clean[key_normalized] = value
                except:
                    labels_clean[key_normalized] = value
            labels_list.append(labels_clean)
        else:
            labels_list.append({})
    
    # Convert to arrays
    images = np.array(images, dtype=np.float32)
    geometries = np.array(geometries, dtype=np.float32)
    
    print(f"✓ Successfully processed {len(images)} samples")
    print(f"  Images shape: {images.shape}")
    print(f"  Geometries shape: {geometries.shape}")
    
    # Save combined file
    print(f"\n[4/5] Saving combined dataset file...")
    output_path = Path(output_file)
    
    np.savez_compressed(
        output_path,
        images=images,
        geometries=geometries,
        elements=elements_list,
        labels=labels_list,
        metadata=metadata_list,
        pair_names=pair_names
    )
    
    file_size = output_path.stat().st_size / (1024**2)
    print(f"✓ Saved: {output_path}")
    print(f"  File size: {file_size:.1f} MB")
    print(f"  Samples: {len(images)}")
    
    # Save analysis files
    print(f"\n[5/5] Saving analysis files...")
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)
    
    # Save labels CSV with normalized column names
    labels_df_normalized = labels_df.reset_index()
    labels_df_normalized.columns = [col.lower() for col in labels_df_normalized.columns]
    labels_df_normalized.to_csv(analysis_dir / "all_labels.csv", index=False)
    print(f"✓ Saved: analysis/all_labels.csv")
    
    # Save geometry data CSV
    geometry_data = []
    for i in range(len(geometries)):
        if len(geometries[i]) == 2:
            geometry_data.append({
                'pair_name': pair_names[i],
                'element_1': elements_list[i][0],
                'element_2': elements_list[i][1],
                'x1': geometries[i][0, 0],
                'y1': geometries[i][0, 1],
                'z1': geometries[i][0, 2],
                'population_1': geometries[i][0, 3],
                'x2': geometries[i][1, 0],
                'y2': geometries[i][1, 1],
                'z2': geometries[i][1, 2],
                'population_2': geometries[i][1, 3],
                'bond_length_ang': metadata_list[i]['bond_length'],
            })
    
    df_geom = pd.DataFrame(geometry_data)
    df_geom.to_csv(analysis_dir / "geometry_data.csv", index=False)
    print(f"✓ Saved: analysis/geometry_data.csv")
    
    # Save comprehensive label summary
    with open(analysis_dir / "labels_detailed_summary.txt", 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TWO-BODY QUANTUM SYSTEMS - COMPREHENSIVE LABEL SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Samples: {len(images)}\n")
        f.write(f"Total Label Columns: {len(labels_df_normalized.columns) - 1}\n\n")
        
        for col in labels_df_normalized.columns:
            if col == 'pair_name':
                continue
            
            values = labels_df_normalized[col]
            non_null = values.dropna()
            
            f.write(f"\n{col}:\n")
            f.write(f"  Coverage: {len(non_null)}/{len(values)} ({100*len(non_null)/len(values):.1f}%)\n")
            
            if len(non_null) > 0 and pd.api.types.is_numeric_dtype(non_null):
                finite = non_null[np.isfinite(non_null)]
                if len(finite) > 0:
                    f.write(f"  Mean: {finite.mean():.6f}\n")
                    f.write(f"  Std: {finite.std():.6f}\n")
                    f.write(f"  Range: [{finite.min():.6f}, {finite.max():.6f}]\n")
    
    print(f"✓ Saved: analysis/labels_detailed_summary.txt")
    
    print("\n" + "=" * 70)
    print("✅ COMPLETE!")
    print("=" * 70)
    print(f"\nDataset saved to: {output_path}")
    print(f"  • Single file: {file_size:.1f} MB")
    print(f"  • Samples: {len(images)}")
    print(f"  • Images: {images.shape}")
    print(f"  • Geometries: {geometries.shape}")
    print(f"  • Labels per sample: 37")
    print(f"\n  Analysis folder:")
    print(f"    - all_labels.csv: All 37 labels in CSV format")
    print(f"    - geometry_data.csv: 3D coordinates")
    print(f"    - labels_detailed_summary.txt: Complete statistics")
    print(f"\n  Use with PyTorch:")
    print(f"    from pytorch_dataset import TwoBodyDataset")
    print(f"    dataset = TwoBodyDataset('{output_path.name}', target_label='e_g_ev')")


if __name__ == "__main__":
    import sys
    
    # Allow custom paths from command line
    if len(sys.argv) > 1:
        raw_data_dir = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "data/dataset_combined.npz"
    else:
        raw_data_dir = "raw_data"
        output_file = "data/dataset_combined.npz"
    
    process_dataset(
        raw_data_dir=raw_data_dir,
        output_file=output_file
    )

