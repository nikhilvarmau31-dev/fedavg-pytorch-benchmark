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
def partition_data(dataset, num_clients=10, iid=True):
    num_samples = len(dataset)
    indices = np.arange(num_samples)

    if iid:
        np.random.shuffle(indices)
        splits = np.array_split(indices, num_clients)
    else:
        # Sort by labels so each client gets skewed class distributions (2 shards per client)
        labels = np.array(dataset.targets)
        sorted_indices = indices[np.argsort(labels)]
        shards = np.array_split(sorted_indices, num_clients * 2)
        shard_ids = np.random.permutation(num_clients * 2)
        splits = [
            np.concatenate([shards[shard_ids[2 * i]], shards[shard_ids[2 * i + 1]]])
            for i in range(num_clients)
        ]

    return [Subset(dataset, split) for split in splits]


# 3. Local Client Training
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


# 6. Main Federated Simulation Loop
def run_experiment(train_dataset, test_loader, iid=True, rounds=15, num_clients=10):
    client_datasets = partition_data(train_dataset, num_clients=num_clients, iid=iid)
    global_model = SimpleCNN().to(device)
    accuracies = []

    mode_name = "IID" if iid else "Non-IID"
    print(f"\n--- Running FedAvg ({mode_name}) ---")

    for r in range(1, rounds + 1):
        local_weights = []
        # Select 50% of clients per round (partial participation)
        selected_clients = np.random.choice(num_clients, size=num_clients // 2, replace=False)

        for cid in selected_clients:
            local_model = copy.deepcopy(global_model)
            w = train_client(local_model, client_datasets[cid], epochs=2)
            local_weights.append(w)

        # Aggregate weights and update global model
        global_weights = federated_average(local_weights)
        global_model.load_state_dict(global_weights)

        acc = evaluate(global_model, test_loader)
        accuracies.append(acc)
        print(f"Round {r:02d}/{rounds:02d} | Test Accuracy: {acc:.2f}%")

    return accuracies


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    iid_acc = run_experiment(train_data, test_loader, iid=True, rounds=15)
    non_iid_acc = run_experiment(train_data, test_loader, iid=False, rounds=15)

    # Plot and save convergence comparison
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 16), iid_acc, marker="o", label="FedAvg (IID Data)")
    plt.plot(range(1, 16), non_iid_acc, marker="s", label="FedAvg (Non-IID Data)")
    plt.title("FedAvg Convergence: IID vs Non-IID Client Data (MNIST)")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Global Test Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fedavg_results.png", dpi=150)
    print("\nSaved plot to fedavg_results.png")
