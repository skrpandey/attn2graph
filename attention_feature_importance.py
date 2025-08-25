
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import argparse

# --------------- 1) Synthetic data (user-provided function) ---------------
def generate_synthetic_data(n_samples=1000, n_features=5):
    X = np.random.randn(n_samples, n_features)
    y = 3 * X[:, 0] + 2 * X[:, 1] + 0.5 * X[:, 2] + 0.1 * np.random.randn(n_samples)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# --------------- 2) Dataset wrapper ---------------
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --------------- 3) Feature-wise Attention Regressor ---------------
class FeatureAttentionRegressor(nn.Module):
    """
    Treat features as a sequence of tokens (length = n_features).
    Each token (scalar feature) is embedded, scored by a learnable query,
    turned into attention weights (softmax), pooled, then regressed to y.
    """
    def __init__(self, n_features: int, d_model: int = 16):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Embed each scalar feature to a d_model-dim token (shared weights across features)
        self.token_embed = nn.Linear(1, d_model)

        # Global query vector for dot-product attention
        self.query = nn.Parameter(torch.randn(d_model))

        # Regression head on the pooled representation
        self.regressor = nn.Linear(d_model, 1)

        # Optional: small initialization for stability
        nn.init.xavier_uniform_(self.token_embed.weight)
        nn.init.zeros_(self.token_embed.bias)
        nn.init.xavier_uniform_(self.regressor.weight)
        nn.init.zeros_(self.regressor.bias)

    def forward(self, X):
        """
        X: (B, F)
        Returns: y_hat (B, 1), attn (B, F)
        """
        B, F = X.shape
        assert F == self.n_features, f"Bad input shape: expected n_features={self.n_features} got {F}"

        # Treat features as tokens: (B, F, 1) -> (B, F, d_model)
        tokens = self.token_embed(X.unsqueeze(-1))        # (B, F, d_model)
        tokens = torch.tanh(tokens)                       # nonlinearity

        # Dot-product with global query -> scores (B, F)
        scores = (tokens @ (self.query / math.sqrt(self.d_model)))  # (B, F)

        # Attention over features
        attn = torch.softmax(scores, dim=1)               # (B, F)

        # Weighted sum of token embeddings -> pooled representation
        context = (attn.unsqueeze(-1) * tokens).sum(dim=1)  # (B, d_model)

        # Regression
        y_hat = self.regressor(context)                   # (B, 1)
        return y_hat, attn

# --------------- 4) Train / Eval utilities ---------------
def attention_entropy(attn: torch.Tensor) -> torch.Tensor:
    """
    attn: (B, F) softmax weights
    Returns average entropy per sample (scalar).
    Minimizing this encourages sharper (peaky) attention.
    """
    eps = 1e-12
    entropy = -(attn * torch.log(attn + eps)).sum(dim=1)  # (B,)
    return entropy.mean()

def train_model(model, train_loader, val_loader, epochs=120, lr=1e-3, lambda_entropy=1e-3, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            y_hat, attn = model(Xb)
            loss = mse(y_hat, yb) + lambda_entropy * attention_entropy(attn)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * Xb.size(0)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for Xv, yv in val_loader:
                Xv = Xv.to(device); yv = yv.to(device)
                y_pred, _ = model(Xv)
                val_loss += mse(y_pred, yv).item() * Xv.size(0)

        train_rmse = math.sqrt(total_loss / len(train_loader.dataset))
        val_rmse = math.sqrt(val_loss / len(val_loader.dataset))

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 10 == 0 or ep == 1 or ep == epochs:
            print(f"Epoch {ep:03d} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val

@torch.no_grad()
def evaluate(model, data_loader, device="cpu"):
    model.eval().to(device)
    mse = nn.MSELoss(reduction="sum")
    total = 0.0
    n = 0
    for Xb, yb in data_loader:
        Xb = Xb.to(device); yb = yb.to(device)
        y_hat, _ = model(Xb)
        total += mse(y_hat, yb).item()
        n += Xb.size(0)
    rmse = math.sqrt(total / n)
    return rmse

# --------------- 5) Feature importance extraction ---------------
@torch.no_grad()
def extract_global_feature_importance(model, data_loader, device="cpu"):
    """
    Returns mean attention over the dataset: shape (n_features,)
    """
    model.eval().to(device)
    num_features = model.n_features
    attn_sum = torch.zeros(num_features, device=device)
    count = 0

    for Xb, _ in data_loader:
        Xb = Xb.to(device)
        _, attn = model(Xb)              # (B, F)
        attn_sum += attn.sum(dim=0)
        count += attn.size(0)

    mean_attn = (attn_sum / count).detach().cpu().numpy()  # (F,)
    return mean_attn

@torch.no_grad()
def extract_per_sample_attention(model, X, device="cpu"):
    """
    X: (B, F) tensor
    Returns attention weights per sample: (B, F)
    """
    model.eval().to(device)
    X = X.to(device)
    _, attn = model(X)
    return attn.detach().cpu().numpy()

# --------------- 6) Main: train, test, and show importance ---------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--features", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--lambda_entropy", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Data
    X, y = generate_synthetic_data(n_samples=args.samples, n_features=args.features)
    dataset = TabularDataset(X, y)

    # Split: 70% train, 15% val, 15% test
    n_train = int(0.7 * len(dataset))
    n_val = int(0.15 * len(dataset))
    n_test = len(dataset) - n_train - n_val
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch*2, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=args.batch*2, shuffle=False)

    # Model
    model = FeatureAttentionRegressor(n_features=args.features, d_model=args.d_model)

    # Train
    model, best_val = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, lambda_entropy=args.lambda_entropy, device="cpu"
    )
    print(f"Best Val RMSE: {best_val:.4f}")

    # Test
    test_rmse = evaluate(model, test_loader, device="cpu")
    print(f"Test RMSE: {test_rmse:.4f}")

    # Global feature importance (mean attention)
    mean_attn = extract_global_feature_importance(model, test_loader, device="cpu")
    ranking = np.argsort(mean_attn)[::-1]

    print("\n--- Global Feature Importance (mean attention over test set) ---")
    for rank, fidx in enumerate(ranking, start=1):
        print(f"{rank:>2}. feature[{fidx}]  mean_attn = {mean_attn[fidx]:.4f}")

    # Compare with ground-truth (normalized abs coefficients)
    true_coefs = np.array([3.0, 2.0, 0.5, 0.0, 0.0])
    gt_importance = true_coefs / true_coefs.sum()
    print("\n Ground-truth normalized importances (by generating function):")
    for i, v in enumerate(gt_importance):
        print(f"feature[{i}]: {v:.4f}")

    # (Optional) Quick correlation sanity check
    corr = np.corrcoef(mean_attn, gt_importance)[0, 1]
    print(f"\n Correlation(mean_attn, ground_truth_importance) = {corr:.4f}")

    # (Optional) Per-sample attention for first few rows
    some_X = X[:5]
    per_sample = extract_per_sample_attention(model, some_X, device="cpu")
    print("\n Per-sample attention for first 5 rows (rows sum to 1):")
    for i in range(per_sample.shape[0]):
        print(f"sample {i}: {np.round(per_sample[i], 4)}")

if __name__ == "__main__":
    main()
