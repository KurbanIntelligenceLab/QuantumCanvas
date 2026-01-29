"""
Improved fusion models for modality ablation experiments.

Key improvements:
1. QuantumShellNetV2: Learnable element embeddings + attention fusion
2. CrossModalFusion: Gated cross-attention between modalities
3. FiLMFusion: Feature-wise Linear Modulation for conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumShellNetV2(nn.Module):
    """
    Improved QuantumShellNet with:
    - Learnable element embeddings (not just summed Z)
    - Cross-attention between image and atom features
    - Residual connections in CNN
    - Batch normalization
    """
    
    def __init__(self, 
                 num_channels: int = 10,
                 element_embed_dim: int = 80,  # ~350K params total
                 cnn_channels: list = [64, 128, 112, 56, 32],
                 attention_heads: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        
        self.element_embed_dim = element_embed_dim
        
        # Learnable element embeddings
        self.element_embedding = nn.Embedding(104, element_embed_dim)
        
        # CNN backbone with residual connections
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        in_ch = num_channels
        for out_ch in cnn_channels:
            self.conv_layers.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2)
            )
            self.bn_layers.append(nn.BatchNorm2d(out_ch))
            in_ch = out_ch
        
        # After 5 stride-2 convs: 32 -> 1x1
        cnn_out_dim = cnn_channels[-1]
        
        # Cross-attention: image features attend to atom embeddings
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=cnn_out_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Project element embeddings to match CNN dim
        self.atom_proj = nn.Linear(element_embed_dim, cnn_out_dim)
        
        # Distance encoding
        self.dist_encoding = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, cnn_out_dim)
        )
        
        # Fusion MLP
        fusion_input_dim = cnn_out_dim * 2 + cnn_out_dim  # attended + pooled atoms + distance
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, images, z, pos, batch):
        B = images.size(0)
        device = images.device
        
        # CNN forward with residual connections
        x = images
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = F.relu(bn(conv(x)))
            x = self.dropout(x)
        
        # Global average pool -> [B, cnn_out_dim]
        cnn_features = x.view(B, -1)
        
        # Get element embeddings and distances for each sample
        atom_features_list = []
        distances = []
        
        for i in range(B):
            mask = batch == i
            z_mol = z[mask]
            pos_mol = pos[mask]
            
            # Element embeddings for both atoms
            z1_emb = self.element_embedding(z_mol[0])
            z2_emb = self.element_embedding(z_mol[1] if len(z_mol) > 1 else z_mol[0])
            
            # Project to CNN dimension
            a1 = self.atom_proj(z1_emb)
            a2 = self.atom_proj(z2_emb)
            atom_features_list.append(torch.stack([a1, a2]))  # [2, cnn_out_dim]
            
            # Compute distance
            if len(pos_mol) > 1:
                dist = (pos_mol[0] - pos_mol[1]).norm().unsqueeze(0)
            else:
                dist = torch.zeros(1, device=device)
            distances.append(dist)
        
        atom_features = torch.stack(atom_features_list)  # [B, 2, cnn_out_dim]
        distances = torch.stack(distances)  # [B, 1]
        
        # Cross-attention: CNN features query atom features
        # Expand CNN features as query: [B, 1, cnn_out_dim]
        query = cnn_features.unsqueeze(1)
        attended, _ = self.cross_attention(query, atom_features, atom_features)
        attended = attended.squeeze(1)  # [B, cnn_out_dim]
        
        # Pool atom features
        atom_pooled = atom_features.mean(dim=1)  # [B, cnn_out_dim]
        
        # Distance encoding
        dist_feat = self.dist_encoding(distances)  # [B, cnn_out_dim]
        
        # Fuse all features
        fused = torch.cat([attended, atom_pooled, dist_feat], dim=-1)
        
        return self.fusion(fused).squeeze(-1)


class GatedFusionBlock(nn.Module):
    """Gated fusion: learns to weight contributions from each modality."""
    
    def __init__(self, dim_a: int, dim_b: int, hidden_dim: int):
        super().__init__()
        
        # Gate computation
        self.gate_net = nn.Sequential(
            nn.Linear(dim_a + dim_b, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # Project both to same dimension
        self.proj_a = nn.Linear(dim_a, hidden_dim)
        self.proj_b = nn.Linear(dim_b, hidden_dim)
    
    def forward(self, feat_a, feat_b):
        # Compute gates
        combined = torch.cat([feat_a, feat_b], dim=-1)
        gates = self.gate_net(combined)  # [B, 2]
        
        # Project features
        proj_a = self.proj_a(feat_a)
        proj_b = self.proj_b(feat_b)
        
        # Weighted combination
        fused = gates[:, 0:1] * proj_a + gates[:, 1:2] * proj_b
        
        return fused, gates


class CrossModalAttention(nn.Module):
    """Bidirectional cross-modal attention."""
    
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.attn_a_to_b = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_b_to_a = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)
        
        self.ffn_a = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        self.ffn_b = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, feat_a, feat_b):
        """
        Args:
            feat_a: [B, N_a, dim] - e.g., image patch features
            feat_b: [B, N_b, dim] - e.g., atom features
        """
        # A attends to B
        attended_a, _ = self.attn_a_to_b(feat_a, feat_b, feat_b)
        feat_a = self.norm_a(feat_a + attended_a)
        feat_a = feat_a + self.ffn_a(feat_a)
        
        # B attends to A
        attended_b, _ = self.attn_b_to_a(feat_b, feat_a, feat_a)
        feat_b = self.norm_b(feat_b + attended_b)
        feat_b = feat_b + self.ffn_b(feat_b)
        
        return feat_a, feat_b


class MultiModalFusionV2(nn.Module):
    """
    Improved multimodal fusion with:
    - Custom 10-channel image encoder (not pretrained ResNet)
    - Cross-modal attention between image patches and atom features
    - Gated fusion to learn modality weighting
    - FiLM conditioning option
    """
    
    def __init__(self,
                 num_channels: int = 10,
                 embed_dim: int = 80,  # ~320K params total
                 num_heads: int = 4,
                 num_cross_layers: int = 1,
                 element_embed_dim: int = 48,
                 dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Image encoder: patch embedding + transformer
        self.patch_size = 4
        num_patches = (32 // self.patch_size) ** 2  # 64 patches
        
        self.patch_embed = nn.Conv2d(num_channels, embed_dim, 
                                      kernel_size=self.patch_size, 
                                      stride=self.patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        
        # Image transformer layers
        self.img_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Atom encoder
        self.element_embedding = nn.Embedding(104, element_embed_dim)
        self.atom_proj = nn.Linear(element_embed_dim + 1, embed_dim)  # +1 for distance
        
        # Cross-modal attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossModalAttention(embed_dim, num_heads, dropout)
            for _ in range(num_cross_layers)
        ])
        
        # Gated fusion
        self.gated_fusion = GatedFusionBlock(embed_dim, embed_dim, embed_dim)
        
        # Final prediction head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
    
    def forward(self, images, z, pos, batch):
        B = images.size(0)
        device = images.device
        
        # Encode images: [B, num_patches, embed_dim]
        img_patches = self.patch_embed(images)  # [B, embed_dim, H/p, W/p]
        img_patches = img_patches.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        img_patches = img_patches + self.pos_embed
        img_features = self.img_transformer(img_patches)  # [B, num_patches, embed_dim]
        
        # Encode atoms: [B, 2, embed_dim]
        atom_features_list = []
        for i in range(B):
            mask = batch == i
            z_mol = z[mask]
            pos_mol = pos[mask]
            
            # Element embeddings
            z1_emb = self.element_embedding(z_mol[0])
            z2_emb = self.element_embedding(z_mol[1] if len(z_mol) > 1 else z_mol[0])
            
            # Distance
            if len(pos_mol) > 1:
                dist = (pos_mol[0] - pos_mol[1]).norm()
            else:
                dist = torch.tensor(0.0, device=device)
            
            # Combine element + distance
            a1 = torch.cat([z1_emb, dist.unsqueeze(0)])
            a2 = torch.cat([z2_emb, dist.unsqueeze(0)])
            atom_features_list.append(torch.stack([a1, a2]))
        
        atom_features = torch.stack(atom_features_list)  # [B, 2, element_embed_dim + 1]
        atom_features = self.atom_proj(atom_features)  # [B, 2, embed_dim]
        
        # Cross-modal attention
        for cross_attn in self.cross_attention_layers:
            img_features, atom_features = cross_attn(img_features, atom_features)
        
        # Pool features
        img_pooled = img_features.mean(dim=1)  # [B, embed_dim]
        atom_pooled = atom_features.mean(dim=1)  # [B, embed_dim]
        
        # Gated fusion
        fused, gates = self.gated_fusion(img_pooled, atom_pooled)
        
        # Prediction
        return self.head(fused).squeeze(-1)


class FiLMConditionedCNN(nn.Module):
    """
    CNN with Feature-wise Linear Modulation (FiLM) conditioning.
    Atomic features modulate the CNN processing of images.
    """
    
    def __init__(self,
                 num_channels: int = 10,
                 element_embed_dim: int = 56,  # ~350K params total
                 cnn_channels: list = [48, 96, 104, 52],
                 dropout: float = 0.2):
        super().__init__()
        
        # Element embeddings
        self.element_embedding = nn.Embedding(104, element_embed_dim)
        
        # CNN layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        # FiLM generators for each conv layer
        self.film_gamma = nn.ModuleList()
        self.film_beta = nn.ModuleList()
        
        conditioning_dim = element_embed_dim * 2 + 1  # Two atoms + distance
        
        in_ch = num_channels
        for out_ch in cnn_channels:
            self.conv_layers.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2)
            )
            self.bn_layers.append(nn.BatchNorm2d(out_ch))
            
            # FiLM: generate scale (gamma) and shift (beta) from conditioning
            self.film_gamma.append(nn.Linear(conditioning_dim, out_ch))
            self.film_beta.append(nn.Linear(conditioning_dim, out_ch))
            
            in_ch = out_ch
        
        # Final layers
        final_spatial = 32 // (2 ** len(cnn_channels))  # After stride-2 convs
        final_dim = cnn_channels[-1] * final_spatial * final_spatial
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, images, z, pos, batch):
        B = images.size(0)
        device = images.device
        
        # Get conditioning from atoms
        conditioning_list = []
        for i in range(B):
            mask = batch == i
            z_mol = z[mask]
            pos_mol = pos[mask]
            
            # Element embeddings
            z1_emb = self.element_embedding(z_mol[0])
            z2_emb = self.element_embedding(z_mol[1] if len(z_mol) > 1 else z_mol[0])
            
            # Distance
            if len(pos_mol) > 1:
                dist = (pos_mol[0] - pos_mol[1]).norm().unsqueeze(0)
            else:
                dist = torch.zeros(1, device=device)
            
            cond = torch.cat([z1_emb, z2_emb, dist])
            conditioning_list.append(cond)
        
        conditioning = torch.stack(conditioning_list)  # [B, conditioning_dim]
        
        # CNN with FiLM modulation
        x = images
        for conv, bn, gamma_net, beta_net in zip(
            self.conv_layers, self.bn_layers, self.film_gamma, self.film_beta
        ):
            x = conv(x)
            x = bn(x)
            
            # FiLM: scale and shift
            gamma = gamma_net(conditioning).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
            beta = beta_net(conditioning).unsqueeze(-1).unsqueeze(-1)
            x = gamma * x + beta
            
            x = F.relu(x)
            x = self.dropout(x)
        
        return self.head(x).squeeze(-1)


def get_improved_model(model_type: str, **kwargs):
    """Factory function for improved models."""
    models = {
        'qsn_v2': QuantumShellNetV2,
        'multimodal_v2': MultiModalFusionV2,
        'film_cnn': FiLMConditionedCNN,
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type](**kwargs)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test parameter counts
    print("=" * 60)
    print("IMPROVED MODEL PARAMETER COUNTS")
    print("=" * 60)
    
    for name in ['qsn_v2', 'multimodal_v2', 'film_cnn']:
        model = get_improved_model(name)
        params = count_parameters(model)
        print(f"  {name:<25} {params:>10,} params")
