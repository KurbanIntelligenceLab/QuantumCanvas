"""
Tabular baseline models for modality ablation experiments.

These models use the SAME underlying information as the 10-channel images
but without spatial structure (pooled/flattened features + MLP/Transformer).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TabularMLPRegressor(nn.Module):
    """
    MLP baseline that uses pooled image features + atomic properties.
    
    This provides a fair comparison: same information as vision models,
    but without convolutional/spatial inductive bias.
    
    Input features:
    - Pooled image channels: mean, std, max, min per channel (10 * 4 = 40D)
    - Element embeddings for both atoms (2 * embed_dim)
    - Bond distance (1D)
    """
    
    def __init__(self, 
                 num_channels: int = 10,
                 element_embed_dim: int = 64,
                 hidden_dims: list = [512, 256, 256, 128],  # ~325K params total
                 dropout: float = 0.2,
                 pool_stats: list = ['mean', 'std', 'max', 'min']):
        super().__init__()
        
        self.num_channels = num_channels
        self.pool_stats = pool_stats
        self.element_embed_dim = element_embed_dim
        
        # Element embedding (supports up to Z=103)
        self.element_embedding = nn.Embedding(104, element_embed_dim)
        
        # Calculate input dimension
        # Pooled image features: num_channels * len(pool_stats)
        # Element embeddings: 2 * element_embed_dim
        # Bond distance: 1
        pooled_dim = num_channels * len(pool_stats)
        self.input_dim = pooled_dim + 2 * element_embed_dim + 1
        
        # Build MLP
        layers = []
        prev_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def pool_images(self, images):
        """
        Pool image channels to get statistical features.
        
        Args:
            images: [B, C, H, W] tensor
        Returns:
            [B, C * num_stats] tensor
        """
        B, C, H, W = images.shape
        features = []
        
        # Flatten spatial dimensions
        flat = images.view(B, C, -1)  # [B, C, H*W]
        
        if 'mean' in self.pool_stats:
            features.append(flat.mean(dim=-1))  # [B, C]
        if 'std' in self.pool_stats:
            features.append(flat.std(dim=-1))   # [B, C]
        if 'max' in self.pool_stats:
            features.append(flat.max(dim=-1)[0])  # [B, C]
        if 'min' in self.pool_stats:
            features.append(flat.min(dim=-1)[0])  # [B, C]
        
        return torch.cat(features, dim=-1)  # [B, C * num_stats]
    
    def forward(self, images, z, pos, batch):
        """
        Args:
            images: [B, 10, 32, 32] - 10-channel quantum images
            z: [N] - Atomic numbers for all atoms in batch
            pos: [N, 3] - Atomic positions
            batch: [N] - Batch assignment
        """
        B = images.size(0)
        device = images.device
        
        # Pool image features
        pooled = self.pool_images(images)  # [B, 40]
        
        # Get element embeddings and bond distances for each sample
        z1_embeds = []
        z2_embeds = []
        distances = []
        
        for i in range(B):
            mask = batch == i
            z_mol = z[mask]
            pos_mol = pos[mask]
            
            # Get embeddings for both atoms
            z1_embeds.append(self.element_embedding(z_mol[0]))
            z2_embeds.append(self.element_embedding(z_mol[1] if len(z_mol) > 1 else z_mol[0]))
            
            # Compute bond distance
            if len(pos_mol) > 1:
                dist = (pos_mol[0] - pos_mol[1]).norm()
            else:
                dist = torch.tensor(0.0, device=device)
            distances.append(dist)
        
        z1_embed = torch.stack(z1_embeds)  # [B, embed_dim]
        z2_embed = torch.stack(z2_embeds)  # [B, embed_dim]
        dist = torch.stack(distances).unsqueeze(-1)  # [B, 1]
        
        # Concatenate all features
        features = torch.cat([pooled, z1_embed, z2_embed, dist], dim=-1)
        
        return self.mlp(features).squeeze(-1)


class TabularTransformer(nn.Module):
    """
    Transformer baseline that treats pooled channel features as tokens.
    
    Each image channel becomes a token with its pooled statistics,
    plus special tokens for element identities and geometry.
    """
    
    def __init__(self,
                 num_channels: int = 10,
                 element_embed_dim: int = 64,
                 d_model: int = 80,  # ~320K params total
                 nhead: int = 4,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_channels = num_channels
        self.d_model = d_model
        
        # Element embedding
        self.element_embedding = nn.Embedding(104, element_embed_dim)
        
        # Project pooled stats (4 per channel) to d_model
        self.channel_proj = nn.Linear(4, d_model)
        
        # Project element + distance to d_model (for 2 element tokens)
        self.atom_proj = nn.Linear(element_embed_dim + 1, d_model)
        
        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding for channels + 2 atoms + CLS
        self.pos_encoding = nn.Parameter(torch.randn(1, num_channels + 3, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, images, z, pos, batch):
        B = images.size(0)
        device = images.device
        
        # Get pooled stats per channel: [B, C, 4]
        flat = images.view(B, self.num_channels, -1)
        channel_stats = torch.stack([
            flat.mean(dim=-1),
            flat.std(dim=-1),
            flat.max(dim=-1)[0],
            flat.min(dim=-1)[0]
        ], dim=-1)  # [B, C, 4]
        
        # Project channel stats to tokens
        channel_tokens = self.channel_proj(channel_stats)  # [B, C, d_model]
        
        # Get atom tokens
        atom_tokens = []
        for i in range(B):
            mask = batch == i
            z_mol = z[mask]
            pos_mol = pos[mask]
            
            # Element embeddings
            z1_emb = self.element_embedding(z_mol[0])
            z2_emb = self.element_embedding(z_mol[1] if len(z_mol) > 1 else z_mol[0])
            
            # Distance from each atom to center (or to each other)
            if len(pos_mol) > 1:
                d1 = (pos_mol[0] - pos_mol[1]).norm().unsqueeze(0)
                d2 = d1.clone()
            else:
                d1 = d2 = torch.zeros(1, device=device)
            
            # Combine element + distance
            atom1 = torch.cat([z1_emb, d1])
            atom2 = torch.cat([z2_emb, d2])
            atom_tokens.append(torch.stack([atom1, atom2]))
        
        atom_tokens = torch.stack(atom_tokens)  # [B, 2, embed_dim + 1]
        atom_tokens = self.atom_proj(atom_tokens)  # [B, 2, d_model]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        
        # Concatenate: [CLS] + channel_tokens + atom_tokens
        tokens = torch.cat([cls_tokens, channel_tokens, atom_tokens], dim=1)  # [B, 1+C+2, d_model]
        
        # Add positional encoding
        tokens = tokens + self.pos_encoding
        
        # Transformer
        tokens = self.transformer(tokens)
        
        # Use CLS token for prediction
        cls_out = tokens[:, 0]
        
        return self.head(cls_out).squeeze(-1)


class VisionOnlyRegressor(nn.Module):
    """
    Pure vision model (no atomic number input) for ablation.
    Uses all 10 channels of the image.
    """
    
    def __init__(self, patch_size=4, embed_dim=96, num_heads=4, num_layers=3):  # ~350K params total
        super().__init__()
        
        self.patch_size = patch_size
        num_patches = (32 // patch_size) ** 2
        
        # Patch embedding for 10 channels
        self.patch_embed = nn.Conv2d(10, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1)
        )
    
    def forward(self, images, z=None, pos=None, batch=None):
        """
        Args:
            images: [B, 10, 32, 32]
            z, pos, batch: ignored (for interface compatibility)
        """
        B = images.size(0)
        
        # Patch embedding
        x = self.patch_embed(images)  # [B, embed_dim, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Use CLS for prediction
        return self.head(x[:, 0]).squeeze(-1)


class GeometryOnlyRegressor(nn.Module):
    """
    Geometry-only baseline (no images, just atomic numbers + positions).
    Simplified SchNet-like architecture.
    """
    
    def __init__(self, hidden_dim=192, num_layers=5, dropout=0.1):  # ~320K params total
        super().__init__()
        
        self.element_embedding = nn.Embedding(104, hidden_dim)
        
        # Simple MLP layers with residual connections
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # Distance encoding
        self.dist_proj = nn.Linear(1, hidden_dim)
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, images, z, pos, batch):
        """
        Args:
            images: ignored
            z: [N] atomic numbers
            pos: [N, 3] positions
            batch: [N] batch assignment
        """
        B = batch.max().item() + 1
        device = z.device
        
        outputs = []
        for i in range(B):
            mask = batch == i
            z_mol = z[mask]
            pos_mol = pos[mask]
            
            # Element embeddings
            h1 = self.element_embedding(z_mol[0])
            h2 = self.element_embedding(z_mol[1] if len(z_mol) > 1 else z_mol[0])
            
            # Apply MLP layers
            for layer in self.layers:
                h1 = h1 + layer(h1)
                h2 = h2 + layer(h2)
            
            # Distance feature
            if len(pos_mol) > 1:
                dist = (pos_mol[0] - pos_mol[1]).norm().unsqueeze(0)
            else:
                dist = torch.zeros(1, device=device)
            dist_feat = self.dist_proj(dist)
            
            # Combine
            combined = torch.cat([h1, h2, dist_feat])
            outputs.append(combined)
        
        outputs = torch.stack(outputs)
        return self.head(outputs).squeeze(-1)


def get_modality_model(model_type: str, **kwargs):
    """Factory function for modality ablation models."""
    models = {
        'tabular_mlp': TabularMLPRegressor,
        'tabular_transformer': TabularTransformer,
        'vision_only': VisionOnlyRegressor,
        'geometry_only': GeometryOnlyRegressor,
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type](**kwargs)
