import ssl, certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 1. Lightweight CNN Model for MNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# 2. Data Partitioning: IID vs Non-IID
def partition_data(dataset, num_clients=10, iid=True, seed=42):
    rng = np.random.default_rng(seed)
    num_samples = len(dataset)
    indices = np.arange(num_samples)

    if iid:
        rng.shuffle(indices)
        splits = np.array_split(indices, num_clients)
    else:
        labels = np.array(dataset.targets)
        sorted_indices = indices[np.argsort(labels)]
        shards = np.array_split(sorted_indices, num_clients * 2)
        shard_ids = rng.permutation(num_clients * 2)
        splits = [
            np.concatenate([shards[shard_ids[2 * i]], shards[shard_ids[2 * i + 1]]])
            for i in range(num_clients)
        ]

    return [Subset(dataset, split) for split in splits]


# 3. Local Client Training — epochs is now a parameter
def train_client(model, client_dataset, epochs=2, lr=0.01, batch_size=32):
    model.train()
    loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    return model.state_dict()


# 4. Server Aggregation (FedAvg)
def federated_average(weights_list):
    avg_weights = copy.deepcopy(weights_list[0])
    for key in avg_weights.keys():
        for i in range(1, len(weights_list)):
            avg_weights[key] += weights_list[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(weights_list))
    return avg_weights


# 5. Global Model Evaluation
def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return 100.0 * correct / total


# 6. Federated Simulation Loop (parameterised by local_epochs and iid)
def run_experiment(train_dataset, test_loader, iid=True, rounds=15,
                   num_clients=10, local_epochs=2, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    client_datasets = partition_data(train_dataset, num_clients=num_clients,
                                     iid=iid, seed=seed)
    global_model = SimpleCNN().to(device)
    accuracies = []

    dist_name = "IID" if iid else "Non-IID"
    label = f"E={local_epochs} ({dist_name})"
    print(f"\n--- FedAvg | {label} ---")

    for r in range(1, rounds + 1):
        local_weights = []
        rng = np.random.default_rng(seed + r)
        selected_clients = rng.choice(num_clients, size=num_clients // 2, replace=False)

        for cid in selected_clients:
            local_model = copy.deepcopy(global_model)
            w = train_client(local_model, client_datasets[cid], epochs=local_epochs)
            local_weights.append(w)

        global_weights = federated_average(local_weights)
        global_model.load_state_dict(global_weights)

        acc = evaluate(global_model, test_loader)
        accuracies.append(acc)
        print(f"  Round {r:02d}/{rounds} | Acc: {acc:.2f}%")

    return accuracies, label


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_data  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    rounds = 15
    results = {}   # label -> list of accuracies

    # ── Ablation: 3 epoch settings × 2 data distributions ──────────────────
    for local_epochs in [1, 2, 5]:
        for iid in [True, False]:
            acc, label = run_experiment(
                train_data, test_loader,
                iid=iid, rounds=rounds,
                local_epochs=local_epochs, seed=42
            )
            results[label] = acc

    # ── Print summary table ─────────────────────────────────────────────────
    print("\n\n========== ABLATION SUMMARY (Final Round Accuracy) ==========")
    print(f"{'Setting':<20} {'Final Acc':>10} {'Peak Acc':>10}")
    print("-" * 44)
    for label, accs in results.items():
        print(f"{label:<20} {accs[-1]:>9.2f}%  {max(accs):>9.2f}%")

    # ── Plot 1: Non-IID drift comparison (the key story) ───────────────────
    rounds_range = range(1, rounds + 1)
    styles = {
        "E=1 (Non-IID)":  ("tab:blue",   "o",  "--"),
        "E=2 (Non-IID)":  ("tab:orange", "s",  "-"),
        "E=5 (Non-IID)":  ("tab:red",    "^",  "-."),
        "E=1 (IID)":      ("tab:cyan",   "o",  "--"),
        "E=2 (IID)":      ("tab:green",  "s",  "-"),
        "E=5 (IID)":      ("tab:purple", "^",  "-."),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for dist, ax, title in [("Non-IID", axes[0], "Client Drift: Non-IID (key ablation)"),
                             ("IID",    axes[1], "Reference: IID (minimal drift)")]:
        for label, accs in results.items():
            if dist in label:
                color, marker, ls = styles[label]
                ax.plot(rounds_range, accs, marker=marker, linestyle=ls,
                        color=color, label=label, linewidth=2, markersize=5)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Communication Rounds")
        ax.set_ylabel("Global Test Accuracy (%)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9)

    plt.suptitle("FedAvg Ablation: Local Epochs (E=1 vs E=2 vs E=5) — Client Drift Study",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("fedavg_ablation.png", dpi=150, bbox_inches="tight")
    print("\nSaved plot → fedavg_ablation.png")

    # ── Plot 2: IID vs Non-IID gap per epoch setting ───────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, e in zip(axes2, [1, 2, 5]):
        iid_key    = f"E={e} (IID)"
        noniid_key = f"E={e} (Non-IID)"
        ax.plot(rounds_range, results[iid_key],    marker="o", color="steelblue",
                label="IID",     linewidth=2)
        ax.plot(rounds_range, results[noniid_key], marker="s", color="tomato",
                label="Non-IID", linewidth=2, linestyle="--")
        gap = results[iid_key][-1] - results[noniid_key][-1]
        ax.set_title(f"E={e}  |  IID−NonIID gap = {gap:.1f}%", fontweight="bold")
        ax.set_xlabel("Communication Rounds")
        ax.set_ylabel("Test Accuracy (%)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9)

    plt.suptitle("IID vs Non-IID Accuracy Gap by Local Epochs",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("fedavg_ablation_gap.png", dpi=150, bbox_inches="tight")
    print("Saved plot → fedavg_ablation_gap.png")
