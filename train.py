import torch
import pickle
import networkx as nx

from glob import glob
from model import SimpleGCN


def evaluate(model, networkx_graphs_dict, torch_graphs_dict, labels_dict):
    model.eval()
    cases = ["val1", "val2", "val3", "val4"]

    print()
    print("="*50)
    for case in cases:
        networkx_graphs = networkx_graphs_dict[case]
        torch_graphs = torch_graphs_dict[case]
        labels = labels_dict[case]
        data_size = len(networkx_graphs)
        cut_size = 0.
        loss = 0.
        for nx_graph, tg_graph, label in zip(networkx_graphs, torch_graphs, labels):
            pred = model(tg_graph).view(-1)
            selected_nodes = torch.where(pred > 0.5)[0].tolist()
            label_onehot = torch.zeros_like(pred)
            label_onehot[label] = 1
            cut_size += nx.cut_size(nx_graph, selected_nodes, weight="weight")
            loss += loss_fn(pred, label_onehot).item()
        print(f'Data: {case} / Cut size: {cut_size/data_size:.4f} / Loss: {loss/data_size:.4f}')
    print("="*50)
    print()


# Import train data
n_data = 5000 # Maximum 50000
networkx_graphs = []
torch_graphs = []
labels = []
train_instances = sorted(glob(f'./datasets_xai606/train/instance_*'))
for train_instance in train_instances[:n_data]:
    with open(train_instance, 'rb') as inp:
        networkx_graph, torch_graph, label = pickle.load(inp)
        networkx_graphs.append(networkx_graph)
        torch_graphs.append(torch_graph)
        labels.append(label)

# Import val data
cases = ["val1", "val2", "val3", "val4"]
nx_graphs_val_dict = {"val1": [], "val2": [], "val3": [], "val4": []}
tg_graphs_val_dict = {"val1": [], "val2": [], "val3": [], "val4": []}
labels_val_dict = {"val1": [], "val2": [], "val3": [], "val4": []}
for case in cases:
    val_instances = sorted(glob(f'./datasets_xai606/{case}/instance_*'))
    for val_instance in val_instances:
        with open(val_instance, 'rb') as inp:
            networkx_graph, torch_graph, label = pickle.load(inp)
            nx_graphs_val_dict[case].append(networkx_graph)
            tg_graphs_val_dict[case].append(torch_graph)
            labels_val_dict[case].append(label)


# Model and hyperparameters
model = SimpleGCN(input_dim=3, hidden_dim=32, output_dim=1, n_mp_layers=3)
lr = 0.001
total_epoch = 100
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# Main loop
for epoch in range(total_epoch):
    evaluate(model, nx_graphs_val_dict, tg_graphs_val_dict, labels_val_dict)
    model.train()
    for torch_graph, label in zip(torch_graphs, labels):
        pred = model(torch_graph).view(-1) # (50, 1) -> (50, )
        label_onehot = torch.zeros_like(pred)
        label_onehot[label] = 1
        loss = loss_fn(pred, label_onehot)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
evaluate(model, nx_graphs_val_dict, tg_graphs_val_dict, labels_val_dict)