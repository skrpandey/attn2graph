import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class GNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GNNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.message_weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.update_weight = nn.Parameter(torch.randn(input_dim + output_dim, output_dim))

    def message_function(self, node_features, adj_matrix, node_idx):
        # Collect information from neighbors
        # adj_matrix[node_idx] is a row vector [num_nodes]
        neighbors = adj_matrix[node_idx].nonzero().squeeze(1)  # Get indices of neighbors
        if len(neighbors) == 0:  # Handle isolated nodes
            return torch.zeros(1, self.output_dim, device=node_features.device)
        neighbor_features = node_features[neighbors]  # [num_neighbors, input_dim]
        messages = neighbor_features @ self.message_weight  # [num_neighbors, output_dim]
        return messages

    def aggregate_function(self, messages):
        # Aggregate messages (e.g., sum over neighbors)
        # messages shape is [num_neighbors, output_dim]
        aggregated_message = torch.sum(messages, dim=0)  # Sum over the neighbor dimension
        return aggregated_message.unsqueeze(0)  # Return as [1, output_dim]

    def update_function(self, node_features, aggregated_message):
        # Update node's own features using aggregated message
        # node_features: [1, input_dim], aggregated_message: [1, output_dim]
        combined = torch.cat([node_features, aggregated_message], dim=1)  # [1, input_dim + output_dim]
        updated_features = combined @ self.update_weight  # [1, output_dim]
        return F.relu(updated_features)

    def forward(self, node_features, adj_matrix):
        # Apply message passing for all nodes
        num_nodes = node_features.shape[0]
        updated_features = torch.zeros(num_nodes, self.output_dim, device=node_features.device)

        for i in range(num_nodes):
            # Message step: Collect from neighbors
            messages = self.message_function(node_features, adj_matrix, i)
            # Aggregate step: Combine neighbor messages
            aggregated = self.aggregate_function(messages)
            # Update step: Update node features
            updated_features[i] = self.update_function(node_features[i].unsqueeze(0), aggregated).squeeze(0)
            
        return updated_features

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.layer1 = GNNLayer(input_dim, hidden_dim)
        self.layer2 = GNNLayer(hidden_dim, output_dim)

    def forward(self, node_features, adj_matrix):
        x = self.layer1(node_features, adj_matrix)
        x = self.layer2(x, adj_matrix)
        return x


class GNNLayerV2(nn.Module):
    """
    A single Graph Neural Network layer implementing the message passing framework.
    """

    def __init__(self, input_dim, output_dim, aggregation='sum'):
        """
        Initialize a GNN layer.

        Args:
            input_dim: Dimension of input node features
            output_dim: Dimension of output node features
            aggregation: Aggregation method ('sum', 'mean', 'max')
        """
        super(GNNLayerV2, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aggregation = aggregation

        # Weights for transforming node features into messages
        self.message_weight = nn.Parameter(torch.Tensor(input_dim, output_dim))

        # Weights for the update function
        self.update_weight = nn.Parameter(torch.Tensor(input_dim + output_dim, output_dim))

        # Initialize parameters
        nn.init.xavier_uniform_(self.message_weight)
        nn.init.xavier_uniform_(self.update_weight)

    def message_function(self, node_features, adj_matrix, node_idx):
        """
        Generate messages from neighbors to the target node.

        Args:
            node_features: Features of all nodes [num_nodes, input_dim]
            adj_matrix: Adjacency matrix [num_nodes, num_nodes]
            node_idx: Index of the target node

        Returns:
            Messages from neighbors [num_neighbors, output_dim]
        """
        # Get indices of neighbors (nodes connected to the target node)
        neighbors = adj_matrix[node_idx].nonzero().squeeze(1)

        # Handle isolated nodes (nodes with no neighbors)
        if len(neighbors) == 0:
            return torch.zeros(1, self.output_dim, device=node_features.device)

        # Get features of neighboring nodes
        neighbor_features = node_features[neighbors]  # [num_neighbors, input_dim]

        # Transform neighbor features into messages
        messages = neighbor_features @ self.message_weight  # [num_neighbors, output_dim]

        return messages

    def aggregate_function(self, messages):
        """
        Aggregate messages from neighbors.

        Args:
            messages: Messages from neighbors [num_neighbors, output_dim]

        Returns:
            Aggregated message [1, output_dim]
        """
        if messages.shape[0] == 0:
            # No messages to aggregate
            return torch.zeros(1, self.output_dim, device=messages.device)

        # Aggregate messages based on the specified method
        if self.aggregation == 'mean':
            aggregated = torch.mean(messages, dim=0, keepdim=True)
        elif self.aggregation == 'max':
            aggregated, _ = torch.max(messages, dim=0, keepdim=True)
        else:  # default: sum
            aggregated = torch.sum(messages, dim=0, keepdim=True)

        return aggregated  # [1, output_dim]

    def update_function(self, node_feature, aggregated_message):
        """
        Update node features using the aggregated message.

        Args:
            node_feature: Feature of the target node [1, input_dim]
            aggregated_message: Aggregated message from neighbors [1, output_dim]

        Returns:
            Updated node feature [1, output_dim]
        """
        # Combine node's own feature with the aggregated message
        combined = torch.cat([node_feature, aggregated_message], dim=1)  # [1, input_dim + output_dim]

        # Transform combined features
        updated = combined @ self.update_weight  # [1, output_dim]

        # Apply non-linearity
        return F.relu(updated)

    def forward(self, node_features, adj_matrix):
        """
        Forward pass for the GNN layer.

        Args:
            node_features: Features of all nodes [num_nodes, input_dim]
            adj_matrix: Adjacency matrix [num_nodes, num_nodes]

        Returns:
            Updated node features [num_nodes, output_dim]
        """
        num_nodes = node_features.shape[0]
        updated_features = torch.zeros(num_nodes, self.output_dim, device=node_features.device)

        # Process each node
        for i in range(num_nodes):
            # 1. Generate messages from neighbors
            messages = self.message_function(node_features, adj_matrix, i)

            # 2. Aggregate messages
            aggregated = self.aggregate_function(messages)

            # 3. Update node feature
            updated_features[i] = self.update_function(
                node_features[i].unsqueeze(0),  # [1, input_dim]
                aggregated
            ).squeeze(0)  # [output_dim]

        return updated_features


class GNNV2(nn.Module):
    """
    Graph Neural Network model with multiple layers.
    """

    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1, aggregation='sum'):
        """
        Initialize a multi-layer GNN.

        Args:
            input_dim: Dimension of input node features
            hidden_dims: List of hidden dimensions for each layer
            output_dim: Dimension of output node features
            dropout: Dropout probability
            aggregation: Aggregation method ('sum', 'mean', 'max')
        """
        super(GNNV2, self).__init__()

        # Create a list to hold all layers
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()

        # Create GNN layers
        for i in range(len(layer_dims) - 1):
            self.layers.append(GNNLayerV2(layer_dims[i], layer_dims[i + 1], aggregation))

        self.dropout = nn.Dropout(dropout)
        self.num_layers = len(self.layers)

    def forward(self, node_features, adj_matrix):
        """
        Forward pass for the GNN.

        Args:
            node_features: Features of all nodes [num_nodes, input_dim]
            adj_matrix: Adjacency matrix [num_nodes, num_nodes]

        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        x = node_features

        # Apply each layer except the last one
        for i in range(self.num_layers - 1):
            x = self.layers[i](x, adj_matrix)
            x = self.dropout(x)

        # Apply the last layer without dropout
        x = self.layers[-1](x, adj_matrix)

        return x



def train_model(model, node_features, adj_matrix, labels, epochs=200, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(node_features, adj_matrix)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    return model

# Example usage
def main():
    # Dummy data: 10 nodes, 4 features per node, 2 classes
    num_nodes, input_dim, hidden_dim, output_dim = 10, 4, 8, 2
    node_features = torch.randn(num_nodes, input_dim)
    adj_matrix = torch.randint(0, 2, (num_nodes, num_nodes)).float()  # Binary adjacency matrix
    labels = torch.randint(0, 2, (num_nodes,))  # Random labels for classification

    # Initialize two-layer GNN
    model = GNN(input_dim, hidden_dim, output_dim)

    # Train the model
    trained_model = train_model(model, node_features, adj_matrix, labels)

    # Print final output
    with torch.no_grad():
        predictions = trained_model(node_features, adj_matrix)
        print("Final predictions shape:", predictions.shape)


def train_gnn_v2(model, node_features, adj_matrix, labels, mask=None,
                 epochs=200, lr=0.01, weight_decay=5e-4, patience=20):
    """
    Train a GNN model.

    Args:
        model: GNN model
        node_features: Features of all nodes [num_nodes, input_dim]
        adj_matrix: Adjacency matrix [num_nodes, num_nodes]
        labels: Node labels [num_nodes]
        mask: Boolean mask for nodes to train on [num_nodes]
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization strength
        patience: Early stopping patience

    Returns:
        Trained model and training history
    """
    # If no mask is provided, use all nodes
    if mask is None:
        mask = torch.ones(labels.shape[0], dtype=torch.bool)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # For early stopping
    best_loss = float('inf')
    no_improve = 0
    best_model_state = None

    # Training history
    history = {'loss': [], 'accuracy': []}

    # Training loop
    for epoch in range(epochs):
        # Set model to training mode
        model.train()

        # Forward pass
        optimizer.zero_grad()
        outputs = model(node_features, adj_matrix)

        # Compute loss on masked nodes
        loss = criterion(outputs[mask], labels[mask])

        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute accuracy
        _, predicted = torch.max(outputs[mask].data, 1)
        accuracy = (predicted == labels[mask]).sum().item() / mask.sum().item()

        # Store metrics
        history['loss'].append(loss.item())
        history['accuracy'].append(accuracy)

        # Print progress
        if epoch % 20 == 0:
            print(f'Epoch {epoch:4d} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.4f}')

        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def evaluate_gnn_v2(model, node_features, adj_matrix, labels, mask=None):
    """
    Evaluate a GNN model.

    Args:
        model: GNN model
        node_features: Features of all nodes [num_nodes, input_dim]
        adj_matrix: Adjacency matrix [num_nodes, num_nodes]
        labels: Node labels [num_nodes]
        mask: Boolean mask for nodes to evaluate on [num_nodes]

    Returns:
        Loss and accuracy
    """
    # If no mask is provided, use all nodes
    if mask is None:
        mask = torch.ones(labels.shape[0], dtype=torch.bool)

    # Set model to evaluation mode
    model.eval()

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Forward pass
    with torch.no_grad():
        outputs = model(node_features, adj_matrix)
        loss = criterion(outputs[mask], labels[mask])

        # Compute accuracy
        _, predicted = torch.max(outputs[mask].data, 1)
        accuracy = (predicted == labels[mask]).sum().item() / mask.sum().item()

    return loss.item(), accuracy


def visualize_training_v2(history):
    """
    Visualize training history.

    Args:
        history: Dictionary containing 'loss' and 'accuracy' lists
    """
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'])
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.show()


def example_node_classification_v2():
    """
    Example of node classification using a GNN.
    """
    # Generate a random graph
    num_nodes = 100
    input_dim = 10
    hidden_dims = [16, 16]
    output_dim = 3  # 3 classes

    # Generate random node features
    node_features = torch.randn(num_nodes, input_dim)

    # Generate random adjacency matrix (with some structure)
    adj_matrix = torch.zeros(num_nodes, num_nodes)

    # Create a block structure in the adjacency matrix
    block_size = num_nodes // output_dim
    for i in range(output_dim):
        start = i * block_size
        end = (i + 1) * block_size if i < output_dim - 1 else num_nodes

        # Connect nodes within the same block with higher probability
        for j in range(start, end):
            for k in range(start, end):
                if j != k and torch.rand(1).item() < 0.3:  # 30% chance of connection within block
                    adj_matrix[j, k] = 1.0

        # Connect nodes between blocks with lower probability
        for j in range(start, end):
            for k in range(num_nodes):
                if k < start or k >= end:  # Different block
                    if torch.rand(1).item() < 0.05:  # 5% chance of connection between blocks
                        adj_matrix[j, k] = 1.0

    # Generate labels based on the block structure
    labels = torch.zeros(num_nodes, dtype=torch.long)
    for i in range(output_dim):
        start = i * block_size
        end = (i + 1) * block_size if i < output_dim - 1 else num_nodes
        labels[start:end] = i

    # Split data into train, validation, and test sets
    indices = torch.randperm(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True

    # Create and train the model
    model = GNNV2(input_dim, hidden_dims, output_dim, dropout=0.2, aggregation='sum')

    # Train the model
    model, history = train_gnn_v2(
        model, node_features, adj_matrix, labels,
        mask=train_mask, epochs=300, lr=0.01, patience=30
    )

    # Evaluate on validation set
    val_loss, val_acc = evaluate_gnn_v2(model, node_features, adj_matrix, labels, val_mask)
    print(f'Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')

    # Evaluate on test set
    test_loss, test_acc = evaluate_gnn_v2(model, node_features, adj_matrix, labels, test_mask)
    print(f'Test - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}')

    # Visualize training history
    visualize_training_v2(history)

    return model, history


if __name__ == "__main__":
    main()
    # example_node_classification_v2()