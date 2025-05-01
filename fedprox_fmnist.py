# fedprox_fashionmnist.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import time
import gc

# 1) SEED SETUP
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

# 2) MODEL DEFINITION
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 3) DATA LOADING
def get_fashion_mnist_dataloaders(num_clients, batch_size=32, subset_size=None, alpha=0.5):
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
    test_ds  = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
    targets  = np.array(train_ds.targets)
    num_classes = len(np.unique(targets))

    # Dirichlet split among clients
    idxs = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        c_idxs = np.where(targets == c)[0]
        np.random.shuffle(c_idxs)
        props = np.random.dirichlet(alpha * np.ones(num_clients))
        props = (props * len(c_idxs)).astype(int)
        diff = len(c_idxs) - props.sum()
        for i in np.random.choice(num_clients, diff, replace=True):
            props[i] += 1
        start = 0
        for i, cnt in enumerate(props):
            idxs[i].extend(c_idxs[start:start+cnt].tolist())
            start += cnt

    if subset_size:
        for i in range(num_clients):
            if len(idxs[i]) > subset_size:
                idxs[i] = np.random.choice(idxs[i], subset_size, replace=False).tolist()

    client_loaders = []
    for i in range(num_clients):
        subset = Subset(train_ds, idxs[i])
        client_loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True,
                                         pin_memory=True, num_workers=2, worker_init_fn=worker_init_fn))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             pin_memory=True, num_workers=2, worker_init_fn=worker_init_fn)
    return client_loaders, test_loader

# 4) EVALUATION
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100 * correct / total

# 5) AGGREGATION (FedAvg)
def fedavg_aggregate(updates):
    agg = {}
    for k in updates[0]:
        if updates[0][k].dtype.is_floating_point:
            agg[k] = torch.mean(torch.stack([u[k] for u in updates]), dim=0)
        else:
            agg[k] = updates[0][k]
    return agg

# 6) LOCAL TRAINING (FedProx)
def local_train_fedprox(model, global_state, loader, device, epochs=5, lr=0.04, mu=0.01):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    global_copy = {k: v.clone().detach().to(device) for k, v in global_state.items()}
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            prox = 0
            for n, p in model.named_parameters():
                prox += 0.5 * mu * torch.sum((p - global_copy[n]) ** 2)
            (loss + prox).backward()
            optimizer.step()
    return model.state_dict()

# 7) FEDERATED LOOP (FedProx)
def federated_learning_fedprox(num_clients=5, rounds=5, epochs=5,
                               batch_size=32, lr=0.03, mu=0.01, alpha=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_loaders, test_loader = get_fashion_mnist_dataloaders(num_clients, batch_size,
                                                                subset_size=None, alpha=alpha)
    global_model = LeNet5().to(device)
    global_state = global_model.state_dict()

    for r in range(rounds):
        updates = []
        for loader in client_loaders:
            local = LeNet5().to(device)
            local.load_state_dict(global_state)
            updates.append(local_train_fedprox(local, global_state, loader, device,
                                               epochs=epochs, lr=lr, mu=mu))
        global_state = fedavg_aggregate(updates)
        global_model.load_state_dict(global_state)
        print(f"[FedProx] Round {r+1} done")
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()

    acc = evaluate(global_model, test_loader, device)
    print(f"FedProx Test Accuracy: {acc:.2f}%")
    return global_model

if __name__ == "__main__":
    set_seed(seed)
    start = time.time()
    federated_learning_fedprox(num_clients=5, rounds=5, epochs=10,
                               batch_size=32, lr=0.03, mu=0.01, alpha=0.1)
    print(f"Total Time: {time.time()-start:.1f}s")
