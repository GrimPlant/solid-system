import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import gc
import time
import random

# ----------------------------
# 1) SEED SETUP AND MODEL DEF
# ----------------------------

seed = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed)

def worker_init_fn(worker_id):
    np.random.seed(seed + worker_id)

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ----------------------------
# 2) DATASET LOADING
# ----------------------------

def get_fashion_mnist_dataloaders(num_clients, batch_size=32, subset_size=None, alpha=0.5):

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

    targets = np.array(list(dataset.targets))
    num_classes = len(np.unique(targets))

    clients_indices = [[] for _ in range(num_clients)]
    for cls in range(num_classes):
        cls_indices = np.where(targets == cls)[0]
        np.random.shuffle(cls_indices)
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        proportions = (proportions * len(cls_indices)).astype(int)
        diff = len(cls_indices) - np.sum(proportions)
        for i in np.random.choice(range(num_clients), diff, replace=True):
            proportions[i] += 1
        start = 0
        for client in range(num_clients):
            num_samples = proportions[client]
            clients_indices[client].extend(cls_indices[start:start + num_samples].tolist())
            start += num_samples

    if subset_size is not None:
        for client in range(num_clients):
            if len(clients_indices[client]) > subset_size:
                clients_indices[client] = np.random.choice(
                    clients_indices[client], subset_size, replace=False
                ).tolist()

    client_datasets = [Subset(dataset, indices) for indices in clients_indices]
    client_loaders = [
        DataLoader(
            subset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2,
            worker_init_fn=worker_init_fn
        )
        for subset in client_datasets
    ]

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2,
        worker_init_fn=worker_init_fn
    )
    return client_loaders, test_loader

def get_test_loader(batch_size=16):
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2,
        worker_init_fn=worker_init_fn
    )
    return test_loader

# ----------------------------
# 3) EVALUATION
# ----------------------------

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100.0 * correct / total
    return accuracy

# ----------------------------
# 4) FISHER SOS COMPUTATION
# ----------------------------

def compute_fim_SOS(model, dataloader, device, max_batches=16):

    model.eval()
    total_g_sq_sum = 0.0
    total_samples = 0
    batch_count = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        model.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
            else:
                grads.append(torch.zeros_like(p).view(-1))
        g = torch.cat(grads)
        g_sq = torch.sum(g ** 2)

        total_g_sq_sum += g_sq.item()
        total_samples += images.size(0)
        batch_count += 1

        if max_batches is not None and batch_count >= max_batches:
            break

    fim_SOS = total_g_sq_sum / (total_samples * 1.0)
    return fim_SOS

# ----------------------------
# 5) AGGREGATION METHODS
# ----------------------------

def fedavg_aggregate(updates):

    aggregated_dict = {}
    for key in updates[0].keys():
        if not updates[0][key].dtype.is_floating_point:
            aggregated_dict[key] = updates[0][key]
        else:
            aggregated_dict[key] = torch.mean(torch.stack([u[key] for u in updates]), dim=0)
    return aggregated_dict

def fim_SOS_aggregate(client_updates, fim_SOSs, temperature=1.0):
    """
    Weigh updates by softmax of SOS(FIM).
    client_updates: list of state_dicts from each client
    fim_SOSs: list of scalar SOS(FIM_i) for each client
    """
    device = list(client_updates[0].values())[0].device
    SOSs_t = torch.tensor(fim_SOSs, device=device, dtype=torch.float)

    weights = torch.softmax(temperature * SOSs_t, dim=0)

    aggregated_dict = {}
    for key in client_updates[0].keys():
        if not client_updates[0][key].dtype.is_floating_point:
            aggregated_dict[key] = client_updates[0][key]
        else:
            agg_value = sum(w * u[key] for w, u in zip(weights, client_updates))
            aggregated_dict[key] = agg_value
    return aggregated_dict

# ----------------------------
# 6) LOCAL TRAINING
# ----------------------------

def local_train(model, dataloader, device, epochs=5, lr=0.4):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
    return model.state_dict()

# ----------------------------
# 7) FEDERATED LEARNING LOOP
# ----------------------------

def federated_learning(
    num_clients=5, rounds=5, epochs=5, batch_size=32, lr=0.03,
    fim_method='fedavg',
    subset_size=None, max_fim_batches=None, alpha=0.5,
    fim_temperature=1.0
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_loaders, _ = get_fashion_mnist_dataloaders(num_clients, batch_size, subset_size, alpha=alpha)
    global_model = LeNet5().to(device)

    for r in range(rounds):
        client_updates = []
        client_SOSs = []

        for i, loader in enumerate(client_loaders):
            local_model = LeNet5().to(device)
            local_model.load_state_dict(global_model.state_dict())

            update = local_train(local_model, loader, device, epochs=epochs, lr=lr)
            client_updates.append(update)

            if fim_method == 'SOS':
                fim_val = compute_fim_SOS(local_model, loader, device, max_batches=max_fim_batches)
                client_SOSs.append(fim_val)

        if fim_method == 'fedavg':
            new_weights = fedavg_aggregate(client_updates)
        elif fim_method == 'SOS':
            new_weights = fim_SOS_aggregate(
                client_updates, client_SOSs, temperature=fim_temperature
            )
        else:
            raise ValueError(f"Unknown fim_method: {fim_method}")

        global_model.load_state_dict(new_weights)
        print(f"Round {r + 1} completed.")
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return global_model

# ----------------------------
# 8) MAIN SCRIPT
# ----------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    test_loader = get_test_loader(batch_size=32)

    print("Running FedAvg...")
    start_time = time.time()
    fedavg_model = federated_learning(
        fim_method='fedavg',
        alpha=0.1,
        rounds=5,
        epochs=30
    )
    fedavg_time = time.time() - start_time
    fedavg_accuracy = evaluate(fedavg_model, test_loader, device)
    print(f"FedAvg Test Accuracy: {fedavg_accuracy:.2f}%")
    print(f"FedAvg Training Time: {fedavg_time:.2f} seconds\n")

    print("Running SOS(FIM)-based weighting...")
    start_time = time.time()
    SOS_model = federated_learning(
        fim_method='SOS',
        alpha=0.1,
        fim_temperature=1,
        rounds=5,
        epochs=30
    )
    SOS_time = time.time() - start_time
    SOS_accuracy = evaluate(SOS_model, test_loader, device)
    print(f"FIM-SOS Weighted Test Accuracy: {SOS_accuracy:.2f}%")
    print(f"FIM-SOS Weighted Training Time: {SOS_time:.2f} seconds\n")