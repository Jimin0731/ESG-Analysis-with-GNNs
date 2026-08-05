import torch
import torch.nn.functional as F
from torch_geometric.datasets import PPI
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch.optim import Adam
from sklearn.metrics import f1_score

# Hyperparameters
hidden_dim = 256
num_layers = 2
sample_sizes = [25, 10]
batch_size = 512
epochs = 10
learning_rate = 0.001

# Dataset
path = './data/PPI'
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')

# Data loaders
train_loader = NeighborLoader(train_dataset[0], num_neighbors=sample_sizes, batch_size=batch_size, shuffle=True)
val_loader = NeighborLoader(val_dataset[0], num_neighbors=sample_sizes, batch_size=batch_size)
test_loader = NeighborLoader(test_dataset[0], num_neighbors=sample_sizes, batch_size=batch_size)

# Model
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='mean'))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr='mean'))
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        return self.lin(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GraphSAGE(train_dataset.num_features, hidden_dim, train_dataset.num_classes).to(device)
optimizer = Adam(model.parameters(), lr=learning_rate)

# Train
def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        loss = F.binary_cross_entropy_with_logits(out, batch.y.to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_nodes
    return total_loss / len(train_loader.dataset)

# Evaluate
@torch.no_grad()
def evaluate(loader):
    model.eval()
    ys, preds = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        pred = (out > 0).float().cpu()
        preds.append(pred)
        ys.append(batch.y.cpu())
    y_true = torch.cat(ys, dim=0).numpy()
    y_pred = torch.cat(preds, dim=0).numpy()
    return f1_score(y_true, y_pred, average='micro')

# Main training loop
for epoch in range(1, epochs + 1):
    loss = train()
    val_f1 = evaluate(val_loader)
    print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Val F1: {val_f1:.4f}")

# Final test score
test_f1 = evaluate(test_loader)
print(f"Test F1: {test_f1:.4f}")

