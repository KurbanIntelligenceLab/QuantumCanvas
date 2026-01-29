# QuantumCanvas: A Multimodal Benchmark for  Visual Learning of Atomic Interactions

**CVPR 2026 Submission under review**

---

## **Abstract**

Despite rapid advances in molecular and materials machine learning, most models lack physical transferability: they fit correlations across whole molecules or crystals rather than learning the quantum interactions between atomic pairs. Yet bonding, charge redistribution, orbital hybridization, and electronic coupling all emerge from these two-body interactions that define local quantum fields in many-body systems.

We introduce **QuantumCanvas**, a large-scale multimodal benchmark that treats two-body quantum systems as foundational units of matter. The dataset spans **2,850 element–element pairs**, each annotated with **18 electronic, thermodynamic, and geometric properties** and paired with **ten-channel image representations** derived from *l*- and *m*-resolved orbital densities, angular field transforms, co-occupancy maps, and charge-density projections. These physically grounded images encode spatial, angular, and electrostatic symmetries without explicit coordinates, providing an interpretable visual modality for quantum learning.

Benchmarking eight architectures across 18 targets, we report MAEs of **0.201 eV** on energy gap with GATv2, **0.265 eV** on HOMO and **0.274 eV** on LUMO with EGNN, and **0.008 Å** on bond length with DimeNet. For energy-related quantities, DimeNet attains **2.27 eV** total-energy MAE and **0.132 eV** repulsive-energy MAE, while a multimodal fusion model achieves a **2.15 eV** Mermin free-energy MAE. Pretraining on **QuantumCanvas** further improves convergence stability and generalization when fine-tuned on **QM9**, **MD17**, and **CrysMTM**.

By unifying orbital physics with vision-based representation learning, **QuantumCanvas** provides a principled and interpretable basis for learning transferable quantum interactions through coupled visual and numerical modalities.

---

## 🚀 Quick Start

### 1. Build Dataset

```bash
python build_dataset.py
```

This creates `dataset_combined.npz` (31.9 MB) with all 2850 samples in one file.

**Custom output:**
```bash
python build_dataset.py /path/to/raw_data my_dataset.npz
```

### 2. Load and Use

**PyTorch (for CNNs/ViTs):**
```python
from pytorch_dataset import TwoBodyDataset
from torch.utils.data import DataLoader

dataset = TwoBodyDataset('dataset_combined.npz', target_label='e_g_ev')
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for images, targets in loader:
    outputs = model(images)  # images: [32, 10, 32, 32]
```

**PyTorch Geometric (for GNNs):**
```python
from pytorch_geometric_dataset import TwoBodyGraphDataset
from torch_geometric.loader import DataLoader

dataset = TwoBodyGraphDataset('dataset_combined.npz', target_labels=['e_g_ev'])
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    outputs = gnn_model(batch.x, batch.edge_index, batch.edge_attr)
```

---

## 📦 What `build_dataset.py` Creates

### Output

```
dataset_combined.npz   → Single file with all 2850 samples (31.9 MB)
├── images:       [2850, 10, 32, 32] - All image tensors
├── geometries:   [2850, 2, 4] - All 3D coordinates
├── elements:     list of 2850 element pairs
├── labels:       list of 2850 label dicts
├── metadata:     list of 2850 metadata dicts
└── pair_names:   list of 2850 system names

analysis/
├── all_labels.csv                 → All 37 labels in CSV format
├── geometry_data.csv              → 3D coordinates for all systems
└── labels_detailed_summary.txt    → Complete label statistics
```

### Access Individual Samples

```python
import numpy as np

# Load entire dataset
data = np.load('dataset_combined.npz', allow_pickle=True)

# Access sample 0
image = data['images'][0]          # [10, 32, 32]
geometry = data['geometries'][0]   # [2, 4]
elements = data['elements'][0]     # ['Ag', 'Al']
labels = data['labels'][0]         # {dict} 37 labels
pair_name = data['pair_names'][0]  # 'Ag_Al'

band_gap = labels['e_g_ev']
```

---

## 📊 Image Channels (10 total)

| Ch | Name | Description |
|----|------|-------------|
| 0-1 | O-Map | Orbital features (radial, angular) |
| 2-3 | RIP-GAF | Rotation-invariant orbitals (s/p, d/f) |
| 4-5 | RIP-MTF | Multipole moments (dipole, quadrupole) |
| 6-7 | COM | Density features (charge, orbital) |
| 8-9 | Q-Image | Charge distribution (positive, negative) |

---

## 🏷️ Labels (37 total)

**Energy (8 float)**
- `total_energy_ev`, `e_homo_ev`, `e_lumo_ev`, `e_g_ev` (band gap)
- `band_energy_ev`, `mermin_free_energy_ev`, `repulsive_energy_ev`, `fermi_level_ev`

**Charge (4 float)**
- `q_absmean`, `q_maxabs`, `q_std`, `total_charge`

**Electronic (10 mixed)**
- `i_ev`, `a_ev`, `chi_ev`, `mu_ev`, `eta_ev` (float)
- `n_levels`, `max_occupancy` (float)
- `softness_evinv`, `electrophilicity_ev` (float)
- `metal_like` (bool: 0/1) 🔵
- `no_virtual_in_basis` (bool: 0/1) 🔵

**Dipole (4 float)**
- `dipole_mag_d`, `dipole_x_d`, `dipole_y_d`, `dipole_z_d`

**Geometric (1 float)**
- `distance_ang` (bond length)

**Convergence (7 mixed)**
- `geom_opt_step`, `scc_last_iter` (float)
- `scc_last_total_elec_eh`, `scc_last_diff_elec`, `scc_last_error` (float)
- `geom_converged`, `scc_converged` (bool: 0/1) 🔵

**System (3 mixed)**
- `n_atoms` (float)
- `system_id_guess` (string)

**Note:** 
- 🔵 = Boolean labels (0/1 values for classification)
- 3D coordinates in `data['geometry']` array, element symbols in `data['elements']`

---

## 💻 Usage Examples

### Simple Regression

```python
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class SimpleDataset(Dataset):
    def __init__(self, data_dir, target='e_g_ev'):
        self.files = sorted(Path(data_dir).glob('*.npz'))
        self.target = target
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        image = torch.from_numpy(data['image']).float()
        target = data['labels'].item()[self.target]
        return image, torch.tensor(target if target else 0.0)
    
    def __len__(self):
        return len(self.files)

# Train on band gap prediction
dataset = SimpleDataset('processed_images', target='e_g_ev')
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### With Geometry Features

```python
class HybridDataset(Dataset):
    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        
        # Image
        image = torch.from_numpy(data['image']).float()
        
        # Geometric features
        geom = data['geometry']
        bond_length = data['metadata'].item()['bond_length']
        
        geom_features = torch.tensor([
            bond_length,
            geom[0, 3] / geom[1, 3],  # population ratio
        ])
        
        # Target
        target = data['labels'].item()[self.target]
        
        return {'image': image, 'geom': geom_features}, target
```

### Using CSV Files

```python
import pandas as pd

# Load labels
df = pd.read_csv('analysis/prediction_labels.csv')

# Load geometry
df_geom = pd.read_csv('analysis/geometry_data.csv')

# Merge
df_full = df.merge(df_geom, on='pair_name')

# Analyze
print(df_full[['pair_name', 'e_g_ev', 'bond_length_ang']].head())
```

---

## 🎯 Common Prediction Tasks

### 1. Band Gap Regression
```python
target = 'e_g_ev'  # Range: [0, 19.4] eV
# Use: Semiconductor applications
```

### 2. Metal Classification
```python
target = 'metal_like'  # Binary: 0 or 1
# Distribution: 74% metal, 26% non-metal
```

### 3. Total Energy Prediction
```python
target = 'total_energy_ev'  # Range: [-305.6, -2.3] eV
# Use: Thermodynamic stability
```

### 4. Multi-Target Learning
```python
targets = ['e_g_ev', 'total_energy_ev', 'dipole_mag_d', 'metal_like']
# Predict multiple properties at once
```

---

## 📈 Dataset Statistics

| Property | Count | Mean | Std | Range |
|----------|-------|------|-----|-------|
| Samples | 2850 | - | - | - |
| Band Gap (eV) | 2850 | 0.47 | 1.19 | [0.0, 19.4] |
| Total Energy (eV) | 2850 | -90.5 | 48.1 | [-305.6, -2.3] |
| Bond Length (Å) | 2850 | 2.58 | 0.66 | [0.7, 5.6] |
| Metal Systems | 2850 | 74% | - | - |

---

## 🔧 Rebuild/Regenerate

### Default (current directory)
```bash
python build_dataset.py
```

### Custom paths
```bash
python build_dataset.py /path/to/raw_data /path/to/output_dir
```

### What it does:
1. ✅ Parses `detailed.out` → orbital populations
2. ✅ Parses `geo_end.xyz` → 3D coordinates  
3. ✅ Creates 10-channel images → `[10, 32, 32]` tensors
4. ✅ Integrates CSV labels → 37 quantum properties
5. ✅ Saves to `dataset_combined.npz` → single file (31.9 MB)
6. ✅ Creates `analysis/` folder → CSVs & summaries

**Processing time:** ~2 minutes for 2850 samples  
**Output:** One file with everything, easy to distribute!

---

## 📁 File Structure

```
.
├── README.md                  ← YOU ARE HERE
├── build_dataset.py           ← Build everything
├── pytorch_dataset.py         ← PyTorch loader
├── pytorch_geometric_dataset.py ← PyTorch Geometric loader
├── check_npz.py               ← Inspect data
│
├── dataset_combined.npz       ← Main dataset (31.9 MB, 2850 samples) ⭐
│
├── raw_data/                  ← Your input data
│   ├── Ag_Al/detailed.out + geo_end.xyz
│   ├── dftb_ptbp_combined.csv
│   └── bond_distances_all.csv
│
└── analysis/                  ← Analysis files
    ├── all_labels.csv
    ├── geometry_data.csv
    └── labels_detailed_summary.txt
```

---

## 🎓 Citation

```bibtex
@dataset{twobody2026,
  title={Two-Body Quantum System Image Dataset},
  year={2026},
  samples={2850},
  image_channels={10},
  labels={37}
}
```

---

## ✅ Validation

- ✅ All 2850 samples processed successfully
- ✅ All labels integrated and verified
- ✅ Bond lengths validated (CSV vs XYZ match)
- ✅ No missing critical data
- ✅ Ready for training

---

## 📖 Label Details

### Energy Labels
- **`e_g_ev`**: HOMO-LUMO gap (band gap) - KEY TARGET
- **`total_energy_ev`**: Total system energy - KEY TARGET
- **`e_homo_ev`**: Highest occupied molecular orbital
- **`e_lumo_ev`**: Lowest unoccupied molecular orbital

### Boolean/Classification Labels
- **`metal_like`** 🔵: Binary metal/non-metal (0=non-metal, 1=metal)
- **`geom_converged`** 🔵: Geometry convergence flag (always 1)
- **`scc_converged`** 🔵: SCC convergence flag (always 1)
- **`no_virtual_in_basis`** 🔵: Virtual orbitals flag

### Regression Targets
All numeric labels can be used as regression targets. See `analysis/all_labels.csv` for the complete list.

---

## 🔍 Verify Data Quality

Check the comprehensive summary to verify all labels:

```bash
cat analysis/labels_detailed_summary.txt
```

This file shows:
- ✅ Coverage for all 48 labels
- ✅ Mean, std, min, max, median for each numeric label
- ✅ Distribution for categorical labels
- ✅ Notes on empty labels

**All labels are lowercase with underscores** (e.g., `e_g_ev`, `total_energy_ev`, `distance_ang`)

**Note:** Geometry coordinates (x, y, z) are in `data['geometry']` array, NOT in labels.

---

## 🔗 PyTorch Geometric Compatibility

**Yes! Your dataset is fully compatible with PyTorch Geometric!**

Each two-body system is a graph with:
- **2 nodes** (atoms)
- **1 edge** (chemical bond)
- **Node features**: Element one-hot + electron population
- **Edge features**: Pooled image channels (10D) + bond vector (4D) = 14D
- **3D positions**: Atomic coordinates
- **Target**: Any of the 37 labels

### Why Use PyG?

✅ **Compare image vs graph approaches** for the same data  
✅ **Hybrid models**: GNN + image features  
✅ **Use 3D geometry** with SchNet, DimeNet, GemNet  
✅ **Message passing** between atoms  
✅ **Benchmark GNNs** against CNNs/ViTs  

### Graph Structure

```
Two-Body System (e.g., Ag-Al):
  Node 0 (Ag): [one-hot Ag, population=11.07]
  Node 1 (Al): [one-hot Al, population=2.93]
  Edge 0→1: [10 image channels (pooled), bond_length, bond_vector]
```

See `pytorch_geometric_dataset.py` for full implementation!

---

## 📤 Releasing as a Dataset

**Recommended release package:**

```
TwoBody-CVPR2026/
├── dataset_combined.npz           (32 MB) ⭐
├── pytorch_dataset.py
├── pytorch_geometric_dataset.py
├── README.md
├── LICENSE
└── analysis/
    ├── all_labels.csv
    ├── geometry_data.csv
    └── labels_detailed_summary.txt
```

**Total size:** ~35 MB

**Upload to:** Zenodo (get DOI), Hugging Face, or GitHub Release

**See `RELEASE_GUIDE.md` for detailed recommendations.**

---

**Questions? Check `analysis/labels_detailed_summary.txt` for complete label statistics.**
