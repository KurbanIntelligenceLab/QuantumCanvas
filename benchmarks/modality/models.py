import torch

from benchmarks.pairs import pair_view
import torch.nn as nn

class TabularMLPRegressor(nn.Module):

    def __init__(self,
                 num_channels: int = 10,
                 element_embed_dim: int = 64,
                 hidden_dims: list = [512, 256, 256, 128],
                 dropout: float = 0.2,
                 pool_stats: list = ['mean', 'std', 'max', 'min']):
        super().__init__()

        self.num_channels = num_channels
        self.pool_stats = pool_stats
        self.element_embed_dim = element_embed_dim

        self.element_embedding = nn.Embedding(104, element_embed_dim)

        pooled_dim = num_channels * len(pool_stats)
        self.input_dim = pooled_dim + 2 * element_embed_dim + 1

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

        B, C, H, W = images.shape
        features = []

        flat = images.view(B, C, -1)

        if 'mean' in self.pool_stats:
            features.append(flat.mean(dim=-1))
        if 'std' in self.pool_stats:
            features.append(flat.std(dim=-1))
        if 'max' in self.pool_stats:
            features.append(flat.max(dim=-1)[0])
        if 'min' in self.pool_stats:
            features.append(flat.min(dim=-1)[0])

        return torch.cat(features, dim=-1)

    def forward(self, images, z, pos, batch):

        B = images.size(0)

        pooled = self.pool_images(images)

        zz, _, dist = pair_view(z, pos, batch, B)
        z1_embed = self.element_embedding(zz[:, 0])
        z2_embed = self.element_embedding(zz[:, 1])
        dist = dist.unsqueeze(-1)

        features = torch.cat([pooled, z1_embed, z2_embed, dist], dim=-1)

        return self.mlp(features).squeeze(-1)

class TabularTransformer(nn.Module):

    def __init__(self,
                 num_channels: int = 10,
                 element_embed_dim: int = 64,
                 d_model: int = 80,
                 nhead: int = 4,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.num_channels = num_channels
        self.d_model = d_model

        self.element_embedding = nn.Embedding(104, element_embed_dim)

        self.channel_proj = nn.Linear(4, d_model)

        self.atom_proj = nn.Linear(element_embed_dim + 1, d_model)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.pos_encoding = nn.Parameter(torch.randn(1, num_channels + 3, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, images, z, pos, batch):
        B = images.size(0)

        flat = images.view(B, self.num_channels, -1)
        channel_stats = torch.stack([
            flat.mean(dim=-1),
            flat.std(dim=-1),
            flat.max(dim=-1)[0],
            flat.min(dim=-1)[0]
        ], dim=-1)

        channel_tokens = self.channel_proj(channel_stats)

        zz, _, dist = pair_view(z, pos, batch, B)
        dist = dist.unsqueeze(-1)
        atom1 = torch.cat([self.element_embedding(zz[:, 0]), dist], dim=-1)
        atom2 = torch.cat([self.element_embedding(zz[:, 1]), dist], dim=-1)
        atom_tokens = torch.stack([atom1, atom2], dim=1)
        atom_tokens = self.atom_proj(atom_tokens)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        tokens = torch.cat([cls_tokens, channel_tokens, atom_tokens], dim=1)

        tokens = tokens + self.pos_encoding

        tokens = self.transformer(tokens)

        cls_out = tokens[:, 0]

        return self.head(cls_out).squeeze(-1)

class VisionOnlyRegressor(nn.Module):

    def __init__(self, patch_size=4, embed_dim=96, num_heads=4, num_layers=3):
        super().__init__()

        self.patch_size = patch_size
        num_patches = (32 // patch_size) ** 2

        self.patch_embed = nn.Conv2d(10, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, images, z=None, pos=None, batch=None):

        B = images.size(0)

        x = self.patch_embed(images)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed

        x = self.transformer(x)

        return self.head(x[:, 0]).squeeze(-1)

class GeometryOnlyRegressor(nn.Module):

    def __init__(self, hidden_dim=192, num_layers=5, dropout=0.1):
        super().__init__()

        self.element_embedding = nn.Embedding(104, hidden_dim)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])

        self.dist_proj = nn.Linear(1, hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, images, z, pos, batch):

        B = batch.max().item() + 1

        zz, _, dist = pair_view(z, pos, batch, B)
        h1 = self.element_embedding(zz[:, 0])
        h2 = self.element_embedding(zz[:, 1])
        for layer in self.layers:
            h1 = h1 + layer(h1)
            h2 = h2 + layer(h2)
        dist_feat = self.dist_proj(dist.unsqueeze(-1))
        return self.head(torch.cat([h1, h2, dist_feat], dim=-1)).squeeze(-1)

def get_modality_model(model_type: str, **kwargs):

    models = {
        'tabular_mlp': TabularMLPRegressor,
        'tabular_transformer': TabularTransformer,
        'vision_only': VisionOnlyRegressor,
        'geometry_only': GeometryOnlyRegressor,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")

    return models[model_type](**kwargs)
