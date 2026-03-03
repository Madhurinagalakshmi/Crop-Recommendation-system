import torch
import torch.nn.functional as F
from pathlib import Path

from src.models.gcn import GCN


def train_model(seed=42):
    # Set random seed
    torch.manual_seed(seed)

    # Load graph
    data_path = Path("data/graph/crop_graph.pt")
    data = torch.load(data_path, weights_only=False)

    input_dim = data.x.shape[1]
    hidden_dim = 64
    output_dim = len(torch.unique(data.y))

    model = GCN(input_dim, hidden_dim, output_dim)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=5e-4
    )

    def train():
        model.train()
        optimizer.zero_grad()

        out = model(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

        return loss.item()

    def evaluate(mask):
        model.eval()
        out = model(data)
        pred = out.argmax(dim=1)
        correct = pred[mask] == data.y[mask]
        acc = int(correct.sum()) / int(mask.sum())
        return acc

    print(f"\nTraining GNN (Seed={seed}) with Early Stopping...\n")

    Path("artifacts").mkdir(exist_ok=True)

    best_val_acc = 0
    patience = 20
    counter = 0

    for epoch in range(1, 301):

        loss = train()

        train_acc = evaluate(data.train_mask)
        val_acc = evaluate(data.val_mask)

        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), "artifacts/gcn_model.pth")
        else:
            counter += 1

        if counter >= patience:
            print("\nEarly stopping triggered.")
            break

    print("\nBest Validation Accuracy:", best_val_acc)

    # Load best model
    model.load_state_dict(torch.load("artifacts/gcn_model.pth"))

    test_acc = evaluate(data.test_mask)
    print("Final Test Accuracy:", test_acc)
    print("Best model saved to artifacts/gcn_model.pth")

    return test_acc   # ✅ THIS WAS MISSING


if __name__ == "__main__":
    seeds = [1, 2, 3, 4, 5]
    results = []

    for s in seeds:
        print(f"\nRunning seed {s}")
        acc = train_model(seed=s)
        results.append(acc)

    mean_acc = sum(results) / len(results)
    print("\nBaseline GCN Mean Test Accuracy:", mean_acc)