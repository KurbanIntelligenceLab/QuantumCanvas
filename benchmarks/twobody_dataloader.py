import torch
from torch_geometric.data import Dataset, Data
import numpy as np
from pathlib import Path

# Atomic numbers mapping
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
    'X': 0,  # Unknown element
}


class TwoBodyDataset(Dataset):
    """PyTorch Geometric Dataset for Two-Body Quantum Systems"""
    
    def __init__(self, npz_path: str, target_label: str = 'e_g_ev', 
                 transform=None, pre_transform=None, verbose=True,
                 normalize_labels=True, normalization_stats=None):
        """
        Args:
            npz_path: Path to dataset_combined.npz
            target_label: Which label to predict
            verbose: Print loading info
            normalize_labels: Whether to normalize labels to [-1, 1] range
            normalization_stats: Dict with 'min' and 'max' for denormalization (for val/test sets)
        """
        self.npz_path = Path(npz_path)
        self.target_label = target_label
        self.verbose = verbose
        self.normalize_labels = normalize_labels
        
        # Load NPZ data
        if verbose:
            print(f"Loading dataset from {self.npz_path}...")
        data = np.load(self.npz_path, allow_pickle=True)
        
        self.geometries = data['geometries']
        self.elements = data['elements']
        self.labels = data['labels']
        self.pair_names = data['pair_names']
        self.images = data['images'] if 'images' in data else None  # [N, 10, 32, 32]
        
        if verbose:
            print(f"  Loaded {len(self.geometries)} samples")
            print(f"  Target property: {target_label}")
            if self.images is not None:
                print(f"  Images shape: {self.images.shape}")
        
        # Filter valid samples
        self.valid_indices = []
        for i in range(len(self.labels)):
            label_dict = self.labels[i]
            if isinstance(label_dict, dict) and target_label in label_dict:
                value = label_dict[target_label]
                if value is not None and not np.isnan(value) and not np.isinf(value):
                    self.valid_indices.append(i)
        
        if verbose:
            print(f"  Valid samples: {len(self.valid_indices)}/{len(self.labels)}")
        
        # Compute or use provided normalization statistics
        if normalize_labels:
            if normalization_stats is None:
                # Compute statistics from this dataset (training set)
                self._compute_normalization_stats()
            else:
                # Use provided statistics (for val/test sets)
                self.label_min = normalization_stats['min']
                self.label_max = normalization_stats['max']
            
            if verbose:
                print(f"  Label range: [{self.label_min:.4f}, {self.label_max:.4f}]")
                print(f"  Normalized to: [-1, 1]")
        else:
            self.label_min = None
            self.label_max = None
        
        super().__init__(None, transform, pre_transform)
    
    def _compute_normalization_stats(self):
        """Compute min and max of labels for normalization"""
        all_labels = []
        for idx in self.valid_indices:
            label_dict = self.labels[idx]
            all_labels.append(float(label_dict[self.target_label]))
        
        all_labels = np.array(all_labels)
        self.label_min = float(np.min(all_labels))
        self.label_max = float(np.max(all_labels))
        
        # Add small epsilon to avoid division by zero
        if abs(self.label_max - self.label_min) < 1e-10:
            self.label_max = self.label_min + 1.0
    
    def get_normalization_stats(self):
        """Get normalization statistics (for use with val/test sets)"""
        return {
            'min': self.label_min,
            'max': self.label_max
        }
    
    def normalize_label(self, value):
        """Normalize a label value to [-1, 1] range"""
        if not self.normalize_labels or self.label_min is None:
            return value
        # Min-max normalization to [-1, 1]
        return 2.0 * (value - self.label_min) / (self.label_max - self.label_min) - 1.0
    
    def denormalize_label(self, normalized_value):
        """Denormalize a label from [-1, 1] back to original range"""
        if not self.normalize_labels or self.label_min is None:
            return normalized_value
        # Inverse of normalization
        return (normalized_value + 1.0) / 2.0 * (self.label_max - self.label_min) + self.label_min
    
    def len(self):
        return len(self.valid_indices)
    
    def get(self, idx):
        """Convert NPZ data to PyTorch Geometric Data object"""
        actual_idx = self.valid_indices[idx]
        
        # Get geometry
        geom = self.geometries[actual_idx]
        pos = torch.tensor(geom[:, :3], dtype=torch.float32)
        
        # Get atomic numbers
        elements = self.elements[actual_idx]
        z = torch.LongTensor([ELEMENT_TO_Z.get(elem, 0) for elem in elements])
        
        # Get target and normalize if enabled
        label_dict = self.labels[actual_idx]
        y_raw = float(label_dict[self.target_label])
        y_normalized = self.normalize_label(y_raw)
        y = torch.FloatTensor([y_normalized])
        
        # Create node features for EGNN (concatenate pos and atomic number)
        x = torch.cat([pos, z.unsqueeze(1).float()], dim=1)  # [n_atoms, 4]
        
        # Get image if available
        image = None
        if self.images is not None:
            image = torch.tensor(self.images[actual_idx], dtype=torch.float32)  # [10, 32, 32]
        
        # Create Data object
        data = Data(
            z=z,
            pos=pos,
            x=x,
            y=y,
            image=image,
            pair_name=self.pair_names[actual_idx]
        )
        
        return data
