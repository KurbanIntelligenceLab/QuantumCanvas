import torch
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet
import numpy as np
import time


def train_epoch(model, loader, optimizer, device, criterion, target_idx=7):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass - SchNet returns graph-level predictions
        out = model(data.z, data.pos, data.batch)
        
        # Compute loss
        loss = criterion(out.squeeze(), data.y[:, target_idx])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, criterion, target_idx=7):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    for data in loader:
        data = data.to(device)
        
        # Forward pass
        out = model(data.z, data.pos, data.batch)
        
        # Compute loss
        loss = criterion(out.squeeze(), data.y[:, target_idx])
        total_loss += loss.item() * data.num_graphs
        
        predictions.append(out.cpu().numpy())
        targets.append(data.y[:, target_idx].cpu().numpy())
    
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    
    mae = np.mean(np.abs(predictions - targets))
    
    return total_loss / len(loader.dataset), mae


def main():
    # Configuration
    config = {
        'batch_size': 32,
        'epochs': 100,
        'lr': 1e-4,
        'weight_decay': 0.0,
        'hidden_channels': 16,
        'num_filters': 16,
        'num_interactions': 6,
        'num_gaussians': 10,
        'cutoff': 5,
        'target': 3,  # U0 (internal energy at 0K)
        'seed': 42
    }
    
    # Target property names
    target_names = [
        'dipole_moment',      # 0: Dipole moment (D)
        'isotropic_polarizability',  # 1: Isotropic polarizability (Bohr^3)
        'homo',               # 2: HOMO energy (eV)
        'lumo',               # 3: LUMO energy (eV)
        'gap',                # 4: HOMO-LUMO gap (eV)
        'electronic_spatial_extent',  # 5: Electronic spatial extent (Bohr^2)
        'zpve',               # 6: Zero point vibrational energy (eV)
        'U0',                 # 7: Internal energy at 0K (eV)
        'U',                  # 8: Internal energy at 298.15K (eV)
        'H',                  # 9: Enthalpy at 298.15K (eV)
        'G',                  # 10: Free energy at 298.15K (eV)
        'Cv',                 # 11: Heat capacity at 298.15K (cal/mol/K)
    ]
    
    print("=" * 70)
    print("QM9 Molecular Property Prediction with SchNet")
    print("=" * 70)
    print(f"\nTarget Property: {target_names[config['target']]} (index {config['target']})")
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load QM9 dataset
    print("Loading QM9 dataset...")
    path = './data/QM9'
    dataset = QM9(path)
    
    print(f"Dataset size: {len(dataset)} molecules")
    print(f"Number of features per node: {dataset.num_features}")
    print(f"Number of target properties: {dataset.num_classes}")
    print(f"Example molecule: {dataset[0]}")
    print()
    
    # Split dataset (110k train, 10k val, 10k test - standard QM9 split)
    train_dataset = dataset[:110000]
    val_dataset = dataset[110000:120000]
    test_dataset = dataset[120000:]
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    print("\nInitializing SchNet model...")
    model = SchNet(
        hidden_channels=config['hidden_channels'],
        num_filters=config['num_filters'],
        num_interactions=config['num_interactions'],
        num_gaussians=config['num_gaussians'],
        cutoff=config['cutoff'],
        readout='mean'
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=10, min_lr=1e-6
    )
    criterion = torch.nn.L1Loss()
    
    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70 + "\n")
    
    best_val_mae = float('inf')
    best_epoch = 0
    
    for epoch in range(1, config['epochs'] + 1):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion, config['target'])
        
        # Evaluate
        val_loss, val_mae = evaluate(model, val_loader, device, criterion, config['target'])
        
        # Learning rate scheduling
        scheduler.step(val_mae)
        
        epoch_time = time.time() - start_time
        
        # Print progress
        print(f"Epoch {epoch:03d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val MAE: {val_mae:.4f} | "
              f"Time: {epoch_time:.2f}s | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'config': config
            }, 'best_schnet_qm9.pt')
            print(f"  → New best model saved! (MAE: {val_mae:.4f})")
        
        # Early stopping
        if epoch - best_epoch > 30:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Test evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation on Test Set")
    print("=" * 70)
    
    # Load best model
    checkpoint = torch.load('best_schnet_qm9.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_mae = evaluate(model, test_loader, device, criterion, config['target'])
    
    print(f"\nBest Epoch: {checkpoint['epoch']}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Target: {target_names[config['target']]}")
    print("\n" + "=" * 70)
    
    return model, test_mae


if __name__ == '__main__':
    # Check if PyTorch Geometric is installed
    try:
        import torch_geometric
        print(f"PyTorch Geometric version: {torch_geometric.__version__}")
    except ImportError:
        print("ERROR: PyTorch Geometric is not installed!")
        print("\nPlease install with:")
        print("  pip install torch-geometric")
        print("  pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.9.0+cpu.html")
        exit(1)
    
    model, mae = main()
    print(f"\n✓ Training complete! Best MAE: {mae:.4f}")

