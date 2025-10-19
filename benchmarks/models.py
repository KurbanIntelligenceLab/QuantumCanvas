import torch
import torch.nn as nn
from torch_geometric.nn import SchNet
from faenet import FAENet
from gotennet import GotenNetWrapper
from gotennet.models.components.layers import CosineCutoff
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class SchNetRegressor(nn.Module):
    def __init__(self, hidden_channels=16, num_filters=16, num_interactions=2, num_gaussians=8, cutoff=5.0):
        super().__init__()
        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout="add"
        )
    
    def forward(self, z, pos, batch):
        # SchNet directly returns the final prediction
        return self.schnet(z, pos, batch)

class FAENetRegressor(nn.Module):
    def __init__(self, cutoff=5.0, hidden_channels=32, num_filters=32, num_interactions=2):
        super().__init__()
        # Override FAENet's default max_Z by monkey-patching after initialization
        self.faenet = FAENet(
            cutoff=cutoff,
            act="silu",
            preprocess="base_preprocess",
            complex_mp=False,
            max_num_neighbors=20,
            num_gaussians=8,
            num_filters=num_filters,
            hidden_channels=hidden_channels,
            tag_hidden_channels=8,
            pg_hidden_channels=8,
            phys_hidden_channels=0,
            phys_embeds=False,  # Disable physics-aware embeddings
            num_interactions=num_interactions,
            mp_type="base",
            graph_norm=True,
            second_layer_MLP=False,
            skip_co="add",
            energy_head=None,
            regress_forces=None,
            force_decoder_type="mlp",
            force_decoder_model_config={"hidden_channels": hidden_channels},
        )
        
        # Extend FAENet's embeddings to support Z=90 (Thorium)
        # FAENet has multiple embedding tables limited to Z=89
        if hasattr(self.faenet, 'embed_block'):
            embed_block = self.faenet.embed_block
            
            # 1. Extend main atomic embedding (self.emb)
            if hasattr(embed_block, 'emb') and isinstance(embed_block.emb, nn.Embedding):
                old_emb = embed_block.emb
                new_emb = nn.Embedding(120, old_emb.embedding_dim)  # Support up to Z=119
                with torch.no_grad():
                    new_emb.weight[:old_emb.num_embeddings] = old_emb.weight
                    # Initialize new entries with small random values
                    new_emb.weight[old_emb.num_embeddings:] = torch.randn(
                        120 - old_emb.num_embeddings, old_emb.embedding_dim
                    ) * 0.01
                embed_block.emb = new_emb
            
            # 2. Extend physics embeddings if they exist
            if hasattr(embed_block, 'phys_emb'):
                phys_emb = embed_block.phys_emb
                
                # Extend period lookup table
                if hasattr(phys_emb, 'period') and len(phys_emb.period) <= 90:
                    import numpy as np
                    # Extend to Z=119, duplicate last values
                    period_arr = phys_emb.period.cpu().numpy() if torch.is_tensor(phys_emb.period) else phys_emb.period
                    period_extended = np.concatenate([period_arr, np.full(120 - len(period_arr), period_arr[-1])])
                    phys_emb.period = torch.tensor(period_extended, dtype=torch.long)
                
                # Extend group lookup table  
                if hasattr(phys_emb, 'group') and len(phys_emb.group) <= 90:
                    import numpy as np
                    group_arr = phys_emb.group.cpu().numpy() if torch.is_tensor(phys_emb.group) else phys_emb.group
                    group_extended = np.concatenate([group_arr, np.full(120 - len(group_arr), group_arr[-1])])
                    phys_emb.group = torch.tensor(group_extended, dtype=torch.long)
                
                # Extend period/group embedding layers
                for attr in ['period_embedding', 'group_embedding']:
                    if hasattr(embed_block, attr):
                        old_layer = getattr(embed_block, attr)
                        if isinstance(old_layer, nn.Embedding):
                            new_layer = nn.Embedding(old_layer.num_embeddings + 10, old_layer.embedding_dim)
                            with torch.no_grad():
                                new_layer.weight[:old_layer.num_embeddings] = old_layer.weight
                            setattr(embed_block, attr, new_layer)
    
    def forward(self, batch):
        # Ensure FAENet inputs are properly set
        batch = self.ensure_faenet_inputs(batch)
        
        # Get FAENet predictions
        outputs = self.faenet(batch)
        
        return outputs
    
    def ensure_faenet_inputs(self, batch):
        # atomic_numbers
        if not hasattr(batch, "atomic_numbers"):
            if hasattr(batch, "z"):
                batch.atomic_numbers = batch.z
            elif hasattr(batch, "element"):
                element_to_z = {"Zn": 30, "O": 8, "Mg": 12}
                batch.atomic_numbers = torch.tensor(
                    [element_to_z[e] for e in batch.element], device=batch.pos.device
                )
            else:
                raise ValueError("Batch must have 'atomic_numbers', 'z', or 'element'.")
        
        # tag: always set to zeros of correct shape and type
        batch.tag = torch.zeros(
            batch.pos.size(0), dtype=torch.long, device=batch.pos.device
        )
        batch.tags = batch.tag  # Ensure both tag and tags exist
        
        if hasattr(batch, "to_data_list"):
            for data in batch.to_data_list():
                if (
                    (not hasattr(data, "tag"))
                    or (data.tag is None)
                    or (not isinstance(data.tag, torch.Tensor))
                ):
                    data.tag = torch.zeros(
                        data.pos.size(0), dtype=torch.long, device=data.pos.device
                    )
                data.tags = data.tag
        
        return batch

class GotenNetRegressor(nn.Module):
    def __init__(self, n_atom_basis=32, n_interactions=2, cutoff=5.0, num_heads=2, n_rbf=4):
        super().__init__()
        self.gotennet = GotenNetWrapper(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            cutoff_fn=CosineCutoff(cutoff),
            num_heads=num_heads,
            n_rbf=n_rbf
        )
        # Add a final regression layer for band gap prediction
        self.regressor = nn.Linear(n_atom_basis, 1)
    
    def forward(self, z, pos, batch):
        # Create input dictionary for GotenNet
        from torch_geometric.data import Data
        data = Data(z=z, pos=pos, batch=batch)
        
        # Get GotenNet embeddings - returns tuple (h, X)
        h, X = self.gotennet(data)
        
        # Use the scalar features h for regression
        # Global mean pooling
        if batch is None:
            x = torch.mean(h, dim=0, keepdim=True)
        else:
            # Use scatter_mean for proper batch handling
            try:
                from torch_scatter import scatter_mean
                x = scatter_mean(h, batch, dim=0)
            except ImportError:
                # Fallback to simple mean if torch_scatter not available
                x = torch.mean(h, dim=0, keepdim=True)
        # Final regression
        return self.regressor(x).squeeze()

class EGNNRegressor(nn.Module):
    def __init__(self, n_layers=3, feats_dim=1, pos_dim=3, m_dim=128, update_coors=True, update_feats=True, norm_feats=True, norm_coors=False, dropout=0.0, coor_weights_clamp_value=2.0):
        super().__init__()
        # Improved EGNN with more parameters
        from torch_geometric.nn import GCNConv, global_mean_pool
        
        # Embedding layer
        self.embedding = nn.Embedding(100, m_dim)
        
        # Multiple GCN layers
        self.convs = nn.ModuleList([
            GCNConv(m_dim, m_dim) for _ in range(n_layers)
        ])
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(m_dim, m_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout if dropout > 0 else 0.1),
            nn.Linear(m_dim // 2, m_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout if dropout > 0 else 0.1),
            nn.Linear(m_dim // 4, 1)
        )
    
    def forward(self, z, pos, batch):
        from torch_geometric.nn import knn_graph
        
        # Create graph structure
        edge_index = knn_graph(pos, k=5, batch=batch)
        
        # Embed atomic numbers
        x = self.embedding(z)
        
        # Apply GCN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
        
        # Global pooling
        if batch is None:
            pooled = torch.mean(x, dim=0, keepdim=True)
        else:
            try:
                from torch_scatter import scatter_mean
                pooled = scatter_mean(x, batch, dim=0)
            except ImportError:
                pooled = torch.mean(x, dim=0, keepdim=True)
        
        # Apply regression head
        return self.regressor(pooled).squeeze()


class GATv2Regressor(nn.Module):
    """Graph Attention Network v2 - Learns attention weights between atoms"""
    def __init__(self, hidden_channels=64, num_layers=3, heads=4, dropout=0.1):
        super().__init__()
        from torch_geometric.nn import GATv2Conv, global_mean_pool
        
        self.embedding = nn.Embedding(100, hidden_channels)
        self.convs = nn.ModuleList([
            GATv2Conv(hidden_channels, hidden_channels, heads=heads, concat=False, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_channels, 1)
        self.global_pool = global_mean_pool
    
    def forward(self, z, pos, batch):
        from torch_geometric.nn import knn_graph
        import torch.nn.functional as F
        
        # Create edges based on k-nearest neighbors
        edge_index = knn_graph(pos, k=5, batch=batch)
        
        # Embed atomic numbers
        x = self.embedding(z)
        
        # Apply GAT layers with attention
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling
        x = self.global_pool(x, batch)
        
        return self.fc(x).squeeze()


class DimeNetRegressor(nn.Module):
    """DimeNet++ - Directional Message Passing Neural Network"""
    def __init__(self, hidden_channels=128, out_channels=1, num_blocks=4, 
                 int_emb_size=64, basis_emb_size=8, out_emb_channels=256,
                 num_spherical=7, num_radial=6, cutoff=5.0, 
                 envelope_exponent=5, num_before_skip=1, num_after_skip=2, num_output_layers=3):
        super().__init__()
        from torch_geometric.nn import DimeNetPlusPlus
        
        self.dimenet = DimeNetPlusPlus(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size,
            out_emb_channels=out_emb_channels,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers
        ).float()  # Ensure float32
    
    def forward(self, z, pos, batch):
        # Ensure float32 for all inputs
        pos = pos.float()
        return self.dimenet(z, pos, batch).squeeze()


class ViTRegressor(nn.Module):
    """Lightweight Vision Transformer for image-based prediction"""
    def __init__(self, patch_size=4, embed_dim=128, num_heads=4, num_layers=4):
        super().__init__()
        
        # Image specs: 32x32 with patch_size=4 gives 8x8=64 patches
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        num_patches = (32 // patch_size) ** 2
        
        # Patch embedding (ensure float32)
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, dtype=torch.float32)
        
        # Positional embedding (ensure float32)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim, dtype=torch.float32))
        
        # Class token (ensure float32)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim, dtype=torch.float32))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1)
        )
    
    def forward(self, images, z=None, pos=None, batch=None):
        """
        Args:
            images: [B, 3, 32, 32] - RGB images
            z, pos, batch: Not used but kept for interface compatibility
        """
        B = images.size(0)
        
        # Patch embedding
        x = self.patch_embed(images)  # [B, embed_dim, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, embed_dim]
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Use class token for prediction
        x = x[:, 0]
        
        return self.head(x).squeeze()


class QuantumShellNet(nn.Module):
    """CNN-based model using images and atomic properties"""
    def __init__(self):
        super().__init__()

        # Convolutional layers - balanced for ~300K parameters (float32)
        self.conv1 = nn.Conv2d(3, 80, kernel_size=3, padding=1, stride=2, dtype=torch.float32)
        self.conv2 = nn.Conv2d(80, 160, kernel_size=3, padding=1, stride=2, dtype=torch.float32)
        self.conv3 = nn.Conv2d(160, 96, kernel_size=3, padding=1, stride=2, dtype=torch.float32)
        self.conv4 = nn.Conv2d(96, 48, kernel_size=3, padding=1, stride=2, dtype=torch.float32)
        self.conv5 = nn.Conv2d(48, 24, kernel_size=3, padding=1, stride=2, dtype=torch.float32)

        # Fully connected layers (float32)
        self.fc1 = nn.Linear(24 * 1 * 1, 200, dtype=torch.float32)  # After 5 stride-2 convs on 32x32: 32->16->8->4->2->1
        self.fc2 = nn.Linear(200 + 3, 100, dtype=torch.float32)
        self.fc3 = nn.Linear(100 + 3, 50, dtype=torch.float32)
        self.fc4 = nn.Linear(50 + 3, 1, dtype=torch.float32)

        self.dropout = nn.Dropout(0.3)
        self.activation = nn.ReLU()

    def forward(self, images, z, pos, batch):
        """
        Args:
            images: [B, 3, 32, 32] - RGB images (converted from 10-channel)
            z: [N] - Atomic numbers for all atoms in batch
            pos: [N, 3] - Positions (not used directly here)
            batch: [N] - Batch assignment for atoms
        """
        # Extract atomic properties for each molecule
        batch_size = images.size(0)
        
        # For each molecule, get atomic properties from the two atoms
        mass_nums = []
        atom_nums = []
        neutron_nums = []
        
        for i in range(batch_size):
            mask = batch == i
            z_mol = z[mask]  # Atomic numbers for this molecule's atoms
            
            # Sum atomic numbers (total protons)
            atom_num = z_mol.sum().float()
            atom_nums.append(atom_num)
            
            # Approximate mass number (A ≈ 2*Z for lighter elements, adjust for heavier)
            # Using standard atomic masses approximation
            mass_num = (z_mol.float() * 2).sum()  # Rough approximation
            mass_nums.append(mass_num)
            
            # Neutrons = Mass - Protons
            neutron_num = mass_num - atom_num
            neutron_nums.append(neutron_num)
        
        mass_num = torch.stack(mass_nums).unsqueeze(1).to(images.device)
        atom_num = torch.stack(atom_nums).unsqueeze(1).to(images.device)
        neutron_num = torch.stack(neutron_nums).unsqueeze(1).to(images.device)
        
        # CNN forward pass
        x = self.activation(self.conv1(images))
        x = self.dropout(x)
        x = self.activation(self.conv2(x))
        x = self.dropout(x)
        x = self.activation(self.conv3(x))
        x = self.dropout(x)
        x = self.activation(self.conv4(x))
        x = self.dropout(x)
        x = self.activation(self.conv5(x))
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        # FC layers with skip connections
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = torch.cat((x, mass_num, atom_num, neutron_num), dim=1)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = torch.cat((x, mass_num, atom_num, neutron_num), dim=1)
        x = self.activation(self.fc3(x))
        x = self.dropout(x)
        x = torch.cat((x, mass_num, atom_num, neutron_num), dim=1)
        x = self.fc4(x)
        
        return x.squeeze()


class MultiModalRegressor(nn.Module):
    def __init__(self, 
                 hidden_channels=16, 
                 num_filters=16, 
                 num_interactions=2, 
                 num_gaussians=8, 
                 cutoff=5.0,
                 image_features_dim=512,
                 mlp_hidden_dims=[256, 128, 64]):
        super().__init__()
        
        # SchNet for molecular data (geometry)
        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout="add"
        )
        
        # ResNet-18 for image data (ensure float32)
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # Remove the final classification layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        # Convert to float32
        self.resnet = self.resnet.float()
        
        # Fusion MLP: combines SchNet features + Image features
        combined_dim = hidden_channels + image_features_dim
        layers = []
        prev_dim = combined_dim
        for hidden_dim in mlp_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.fusion_mlp = nn.Sequential(*layers)
        
    def forward(self, z, pos, batch, images):
        # Get SchNet features - need to extract before final readout
        from torch_geometric.nn import radius_graph
        from torch_scatter import scatter_mean
        
        # Build graph
        edge_index = radius_graph(pos, r=self.schnet.cutoff, batch=batch, max_num_neighbors=32)
        
        # Compute edge attributes (distances and RBF expansion) like SchNet does
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.schnet.distance_expansion(edge_weight)
        
        # Initial atom embeddings
        h = self.schnet.embedding(z)
        
        # Apply interaction blocks with proper edge_attr
        for interaction in self.schnet.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
        
        # Global pooling to get graph-level features
        schnet_features = scatter_mean(h, batch, dim=0)  # [B, hidden_channels]
        
        # Get ResNet features for image data
        image_features = self.resnet(images)
        image_features = image_features.view(image_features.size(0), -1)  # [B, 512]
        
        # Concatenate both modalities
        combined = torch.cat([schnet_features, image_features], dim=1)  # [B, hidden_channels + 512]
        
        # Final prediction through fusion MLP
        output = self.fusion_mlp(combined)
        
        return output.squeeze()
    
    def get_modality_contributions(self, z, pos, batch, images):
        """
        Get individual modality features for interpretability
        
        Returns:
            Dictionary with features from each modality
        """
        schnet_features = self.schnet(z, pos, batch)
        image_features = self.resnet(images)
        image_features = image_features.view(image_features.size(0), -1)
        
        return {
            'gnn_features': schnet_features,
            'image_features': image_features
        }

# Model factory function
def get_model(model_type, **kwargs):
    """Get a model by type"""
    models = {
        'schnet': SchNetRegressor,
        'faenet': FAENetRegressor,
        'gotennet': GotenNetRegressor,
        'egnn': EGNNRegressor,
        'gatv2': GATv2Regressor,
        'dimenet': DimeNetRegressor,
        'vit': ViTRegressor,
        'quantumshellnet': QuantumShellNet,
        'multimodal': MultiModalRegressor
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type](**kwargs) 