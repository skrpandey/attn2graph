import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric_temporal.nn.recurrent import GConvGRU
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Synthetic data generation
def generate_synthetic_joint_data(num_samples=1000, num_joints=15, timesteps=10, feature_dim=3):
    """
    Generate synthetic joint movement data.
    - num_samples: Number of sequences
    - num_joints: Number of joints (nodes) in the skeleton
    - timesteps: Number of time steps per sequence
    - feature_dim: Dimensions per joint (x, y, z coordinates)
    Returns: node_features, edge_index, targets
    """
    # Define a simple skeleton graph (edges between joints)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 1, 5, 5, 6, 6, 7, 1, 8, 8, 9, 9, 10, 1, 11, 11, 12, 12, 13, 13, 14],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 1, 6, 5, 7, 6, 8, 1, 9, 8, 10, 9, 11, 1, 12, 11, 13, 12, 14, 13]
    ], dtype=torch.long).t().contiguous()

    # Generate synthetic joint positions (x, y, z) with smooth motion
    node_features = []
    targets = []
    for _ in range(num_samples):
        # Random initial positions
        seq = np.random.randn(timesteps, num_joints, feature_dim) * 0.1
        # Add smooth motion (sinusoidal patterns)
        for t in range(timesteps):
            seq[t] += np.sin(t * 0.5 + np.random.randn() * 0.1) * 0.5
        node_features.append(seq[:-1])  # Input: t=0 to t=T-2
        targets.append(seq[1:])  # Target: t=1 to t=T-1
    node_features = torch.tensor(node_features,
                                 dtype=torch.float)  # (num_samples, timesteps-1, num_joints, feature_dim)
    targets = torch.tensor(targets, dtype=torch.float)  # (num_samples, timesteps-1, num_joints, feature_dim)
    return node_features, edge_index, targets


# ST-GAT Model
class STGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(STGAT, self).__init__()
        # Spatial attention with GAT
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
        # Temporal processing with GConvGRU
        self.temporal = GConvGRU(hidden_channels * heads, hidden_channels, K=2)
        # Output layer
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # x: (batch_size, timesteps, num_nodes, in_channels)
        batch_size, timesteps, num_nodes, _ = x.size()
        x_out = []

        for t in range(timesteps):
            # Spatial attention
            x_t = x[:, t, :, :]  # (batch_size, num_nodes, in_channels)
            x_t = F.relu(self.gat1(x_t, edge_index))
            x_t = F.relu(self.gat2(x_t, edge_index))
            x_out.append(x_t)

        # Stack temporal data
        x_out = torch.stack(x_out, dim=1)  # (batch_size, timesteps, num_nodes, hidden_channels)

        # Temporal processing
        x_out = self.temporal(x_out, edge_index)  # (batch_size, timesteps, num_nodes, hidden_channels)

        # Output prediction
        x_out = self.linear(x_out)  # (batch_size, timesteps, num_nodes, out_channels)
        return x_out


# Training and validation function
def train_and_validate():
    # Generate synthetic data
    node_features, edge_index, targets = generate_synthetic_joint_data(
        num_samples=1000, num_joints=15, timesteps=10, feature_dim=3
    )

    # Split into train and validation (80-20 split)
    train_size = int(0.8 * len(node_features))
    train_features, val_features = node_features[:train_size], node_features[train_size:]
    train_targets, val_targets = targets[:train_size], targets[train_size:]

    # Move to device
    train_features = train_features.to(device)
    val_features = val_features.to(device)
    train_targets = train_targets.to(device)
    val_targets = val_targets.to(device)
    edge_index = edge_index.to(device)

    # Initialize model, optimizer, and loss
    model = STGAT(in_channels=3, hidden_channels=16, out_channels=3, heads=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training loop
    num_epochs = 50
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_features, edge_index)
        loss = criterion(out, train_targets)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(val_features, edge_index)
            val_loss = criterion(val_out, val_targets)
            val_losses.append(val_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('loss_plot.png')
    plt.close()

    # Calculate final RMSE
    model.eval()
    with torch.no_grad():
        val_out = model(val_features, edge_index)
        val_out = val_out.cpu().numpy().reshape(-1)
        val_targets_np = val_targets.cpu().numpy().reshape(-1)
        rmse = np.sqrt(mean_squared_error(val_out, val_targets_np))
        print(f'Final Validation RMSE: {rmse:.4f}')


# Run the training and validation
if __name__ == '__main__':
    train_and_validate()