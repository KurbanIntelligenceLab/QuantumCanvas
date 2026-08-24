import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Data as PyGData
except ImportError:
    PyGData = None

class LabelNormalizer:

    def __init__(self, method="standard"):
        self.method = method
        self.means = None
        self.stds = None
        self.mins = None
        self.maxs = None
        self.fitted = False

    def fit(self, labels):

        if self.method == "standard":
            self.means = np.mean(labels, axis=0)
            self.stds = np.std(labels, axis=0)

            self.stds = np.where(self.stds == 0, 1.0, self.stds)
        elif self.method == "minmax":
            self.mins = np.min(labels, axis=0)
            self.maxs = np.max(labels, axis=0)

            ranges = self.maxs - self.mins
            ranges = np.where(ranges == 0, 1.0, ranges)
            self.maxs = self.mins + ranges
        self.fitted = True

    def transform(self, labels):

        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transforming")

        if self.method == "standard":
            return (labels - self.means) / self.stds
        elif self.method == "minmax":
            return (labels - self.mins) / (self.maxs - self.mins)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

    def inverse_transform(self, normalized_labels):

        if not self.fitted:
            raise ValueError("Normalizer must be fitted before inverse transforming")

        if self.method == "standard":
            return normalized_labels * self.stds + self.means
        elif self.method == "minmax":
            return normalized_labels * (self.maxs - self.mins) + self.mins
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

class RegressionLoader(Dataset):

    def __init__(
        self,
        label_dir: str,
        temperature_filter: Optional[Callable[[int], bool]] = None,
        transform: Optional[Callable] = None,
        modalities: Optional[List[str]] = None,
        max_rotations: Optional[int] = None,
        as_pyg_data: bool = False,
        normalize_labels: bool = False,
        normalization_method: str = "standard",
        fit_normalizer_on_data: bool = False,
    ):
        self.label_dir = label_dir
        self.temperature_filter = temperature_filter
        self.transform = transform
        self.modalities = modalities or ["image", "xyz", "text", "element"]
        self.max_rotations = max_rotations
        self.as_pyg_data = as_pyg_data
        self.normalize_labels = normalize_labels
        self.normalization_method = normalization_method
        self.fit_normalizer_on_data = fit_normalizer_on_data
        self.label_data = self._load_label_data()
        self.data = self._prepare_dataset()

        self.normalizer = None
        if self.normalize_labels:
            self.normalizer = LabelNormalizer(method=self.normalization_method)
            if self.fit_normalizer_on_data:
                self._fit_normalizer()

    def _fit_normalizer(self):

        all_labels = []
        for entry in self.data:
            temp = entry["temperature"]
            phase = entry["phase"]
            label_dict = self.label_data[phase][temp]
            label = [
                label_dict["HOMO"],
                label_dict["LUMO"],
                label_dict["Eg"],
                label_dict["Ef"],
                label_dict["Et"],
                label_dict["Eta"],
                label_dict["disp"],
                label_dict["vol"],
                label_dict["bond"],
            ]
            all_labels.append(label)

        all_labels = np.array(all_labels)
        self.normalizer.fit(all_labels)
        print(
            f"Fitted normalizer with method: {self.normalization_method} on {len(all_labels)} samples"
        )

    def set_normalizer(self, normalizer: LabelNormalizer):

        self.normalizer = normalizer
        self.normalize_labels = True

    def _load_label_data(self) -> Dict[str, Dict[int, Dict[str, float]]]:

        phases = ["anatase", "brookite", "rutile"]
        label_data = {phase: {} for phase in phases}

        csv_file = os.path.join(self.label_dir, "labels.csv")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            polymorph = row["Polymorph"].lower()
            temp_str = row["Temperature"]
            temp = (
                int(temp_str.replace("K", ""))
                if "K" in temp_str
                else int(float(temp_str))
            )
            parameter = row["Parameter"]
            value = float(row["Value"])

            if temp not in label_data[polymorph]:
                label_data[polymorph][temp] = {}

            param_mapping = {
                "HOMO": "HOMO",
                "LUMO": "LUMO",
                "Eg": "Eg",
                "Ef": "Ef",
                "Et": "Et",
                "Eta": "Eta",
                "disp": "disp",
                "vol": "vol",
                "bond": "bond",
            }

            if parameter in param_mapping:
                label_data[polymorph][temp][param_mapping[parameter]] = value

        return label_data

    def _get_available_rotations(self, temp_dir: str) -> Dict[str, List[int]]:

        available_rotations = {}
        if "image" in self.modalities:
            images_dir = os.path.join(temp_dir, "images")
            if os.path.isdir(images_dir):
                image_files = [f for f in os.listdir(images_dir) if f.endswith(".png")]
                image_rotations = []
                for f in image_files:
                    if f.startswith("rot_") and f.endswith(".png"):
                        try:
                            rot_num = int(f[4:-4])
                            image_rotations.append(rot_num)
                        except ValueError:
                            continue
                available_rotations["image"] = sorted(image_rotations)
        if "xyz" in self.modalities or "element" in self.modalities:
            xyz_dir = os.path.join(temp_dir, "xyz")
            if os.path.isdir(xyz_dir):
                xyz_files = [f for f in os.listdir(xyz_dir) if f.endswith(".xyz")]
                xyz_rotations = []
                for f in xyz_files:
                    if f.startswith("rot_") and f.endswith(".xyz"):
                        try:
                            rot_num = int(f[4:-4])
                            xyz_rotations.append(rot_num)
                        except ValueError:
                            continue
                available_rotations["xyz"] = sorted(xyz_rotations)
        if "text" in self.modalities:
            text_dir = os.path.join(temp_dir, "text")
            if os.path.isdir(text_dir):
                text_files = [f for f in os.listdir(text_dir) if f.endswith(".txt")]
                text_rotations = []
                for f in text_files:
                    if f.startswith("rot_") and f.endswith(".txt"):
                        try:
                            rot_num = int(f[4:-4])
                            text_rotations.append(rot_num)
                        except ValueError:
                            continue
                available_rotations["text"] = sorted(text_rotations)
        return available_rotations

    def _prepare_dataset(self) -> List[Dict[str, Any]]:

        data = []
        phases = ["anatase", "brookite", "rutile"]
        for phase in phases:
            for temp in range(0, 1001, 50):
                if self.temperature_filter and not self.temperature_filter(temp):
                    continue
                temp_dir = os.path.join(self.label_dir, phase, f"{temp}K")
                if not os.path.exists(temp_dir):
                    continue
                available_rotations = self._get_available_rotations(temp_dir)
                if not available_rotations:
                    continue
                common_rotations = set(available_rotations[self.modalities[0]])
                for modality in self.modalities[1:]:
                    if modality in available_rotations:
                        common_rotations = common_rotations.intersection(
                            set(available_rotations[modality])
                        )
                if not common_rotations:
                    continue
                common_rotations = sorted(list(common_rotations))
                if self.max_rotations is not None:
                    common_rotations = common_rotations[: self.max_rotations]
                for rotation in common_rotations:
                    entry = {
                        "phase": phase,
                        "temperature": temp,
                        "rotation": rotation,
                    }
                    if "image" in self.modalities:
                        entry["image_path"] = os.path.join(
                            temp_dir, "images", f"rot_{rotation}.png"
                        )
                    if "xyz" in self.modalities or "element" in self.modalities:
                        entry["xyz_path"] = os.path.join(
                            temp_dir, "xyz", f"rot_{rotation}.xyz"
                        )
                    if "text" in self.modalities:
                        entry["text_path"] = os.path.join(
                            temp_dir, "text", f"rot_{rotation}.txt"
                        )
                    data.append(entry)
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        entry = self.data[idx]
        result = {}

        if self.as_pyg_data and set(self.modalities) == {"xyz", "element"}:
            if PyGData is None:
                raise ImportError("torch_geometric is required for as_pyg_data=True")
            with open(entry["xyz_path"], "r", encoding="utf-8") as f:
                xyz_lines = f.readlines()[2:]
            element_symbols = []
            xyz_coords = []
            for line in xyz_lines:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                elem, x, y, z = parts
                element_symbols.append(elem)
                xyz_coords.append([float(x), float(y), float(z)])
            element_to_idx = {"Ti": 0, "O": 1}
            z = torch.tensor(
                [element_to_idx[el] for el in element_symbols], dtype=torch.long
            )
            pos = torch.tensor(xyz_coords, dtype=torch.float)
            temp = entry["temperature"]
            phase = entry["phase"]
            label_dict = self.label_data[phase][temp]
            y = torch.tensor(
                [
                    label_dict["HOMO"],
                    label_dict["LUMO"],
                    label_dict["Eg"],
                    label_dict["Ef"],
                    label_dict["Et"],
                    label_dict["Eta"],
                    label_dict["disp"],
                    label_dict["vol"],
                    label_dict["bond"],
                ],
                dtype=torch.float,
            )

            if self.normalize_labels and self.normalizer is not None:
                y = torch.tensor(
                    self.normalizer.transform(y.numpy().reshape(1, -1)).flatten(),
                    dtype=torch.float,
                )

            data = PyGData(z=z, pos=pos)
            data.y = y.unsqueeze(0)
            return data

        if "image" in self.modalities:
            image = Image.open(entry["image_path"]).convert("RGB")
            if self.transform:
                image = self.transform(image)
            result["image"] = image
        if "xyz" in self.modalities or "element" in self.modalities:
            with open(entry["xyz_path"], "r", encoding="utf-8") as f:
                xyz_lines = f.readlines()[2:]
            element_symbols = []
            xyz_coords = []
            for line in xyz_lines:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                elem, x, y, z = parts
                element_symbols.append(elem)
                xyz_coords.append([float(x), float(y), float(z)])
            if "xyz" in self.modalities:
                xyz_tensor = torch.tensor(xyz_coords, dtype=torch.float)
                result["xyz"] = xyz_tensor
            if "element" in self.modalities:
                result["element"] = element_symbols
        if "text" in self.modalities:
            with open(entry["text_path"], "r", encoding="utf-8") as f:
                text_data = f.read()
            result["text"] = text_data
        temp = entry["temperature"]
        phase = entry["phase"]
        label_dict = self.label_data[phase][temp]
        regression_label = torch.tensor(
            [
                label_dict["HOMO"],
                label_dict["LUMO"],
                label_dict["Eg"],
                label_dict["Ef"],
                label_dict["Et"],
                label_dict["Eta"],
                label_dict["disp"],
                label_dict["vol"],
                label_dict["bond"],
            ],
            dtype=torch.float,
        )

        if self.normalize_labels and self.normalizer is not None:
            regression_label = torch.tensor(
                self.normalizer.transform(
                    regression_label.numpy().reshape(1, -1)
                ).flatten(),
                dtype=torch.float,
            )

        result["regression_label"] = regression_label

        result["temperature"] = entry["temperature"]
        result["phase"] = entry["phase"]
        result["rotation"] = entry["rotation"]

        return result
