"""QuantumCanvas: a multimodal benchmark for learning two-body quantum interactions."""

from quantumcanvas.constants import (
    BENCHMARK_TARGETS,
    CHANNELS,
    DEFAULT_DATASET_PATH,
    ELEMENT_TO_Z,
)
from quantumcanvas.data import TwoBodyDataset, TwoBodyGraphDataset, batch_images, download, load_npz

__version__ = "1.0.1"

__all__ = [
    "BENCHMARK_TARGETS",
    "CHANNELS",
    "DEFAULT_DATASET_PATH",
    "ELEMENT_TO_Z",
    "TwoBodyDataset",
    "TwoBodyGraphDataset",
    "batch_images",
    "download",
    "load_npz",
]
