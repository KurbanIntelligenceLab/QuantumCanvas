"""
Modality Ablation Experiments for QuantumCanvas Rebuttal.

This package contains experiments to demonstrate that the image modality
provides useful signal beyond element identity priors.

Experiments:
1. train_modality_comparison.py - Tabular vs Vision vs Fusion comparison
2. element_shuffle_ablation.py - Element ID shuffle/mask ablation
3. ood_composition_split.py - Out-of-distribution composition generalization

Models (Baselines):
- TabularMLPRegressor: Pooled image features + atomic info (no spatial structure)
- TabularTransformer: Pooled features as tokens + transformer
- VisionOnlyRegressor: Pure vision (no element IDs)
- GeometryOnlyRegressor: Only Z + positions (no images)

Models (Improved Fusion):
- QuantumShellNetV2: CNN + learnable element embeddings + cross-attention
- MultiModalFusionV2: ViT + atom encoder + cross-modal attention + gated fusion
- FiLMConditionedCNN: CNN with Feature-wise Linear Modulation from atoms
"""

from .models import (
    TabularMLPRegressor,
    TabularTransformer,
    VisionOnlyRegressor,
    GeometryOnlyRegressor,
    get_modality_model,
)

from .improved_models import (
    QuantumShellNetV2,
    MultiModalFusionV2,
    FiLMConditionedCNN,
    get_improved_model,
)

__all__ = [
    # Baseline models
    'TabularMLPRegressor',
    'TabularTransformer', 
    'VisionOnlyRegressor',
    'GeometryOnlyRegressor',
    'get_modality_model',
    # Improved models
    'QuantumShellNetV2',
    'MultiModalFusionV2',
    'FiLMConditionedCNN',
    'get_improved_model',
]
