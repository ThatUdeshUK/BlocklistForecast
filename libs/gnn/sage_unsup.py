import torch
import torch.nn.functional as F
from torch_cluster import random_walk

from torch_geometric.loader import NeighborSampler as RawNeighborSampler
from torch_geometric.nn import SAGEConv

EPS = 1e-15


class NegativeSampler(RawNeighborSampler):
    def sample(self, batch):
        batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        # For each node in `batch`, we sample a direct neighbor (as positive
        # example) and a random node (as negative example):
        pos_batch = random_walk(row, col, batch, walk_length=1,
                                coalesced=False)[:, 1]

        neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(), ),
                                  dtype=torch.long)

        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        return super().sample(batch)


class SAGE(torch.nn.Module):
    def __init__(self, data, in_channels, hidden_channels, num_layers, lr=0.01, dropout=0.5):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_layers = num_layers
        self.data = data.to(self.device)
        
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        self.dropout = dropout
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
    def train_epoch(self, train_loader):
        x = self.data.x

        self.train()

        total_loss = 0
        for batch_size, n_id, adjs in train_loader:
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            adjs = [adj.to(self.device) for adj in adjs]
            self.optimizer.zero_grad()

            out = self.forward(x[n_id], adjs)
            out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

            pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
            neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
            loss = -pos_loss - neg_loss
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss) * out.size(0)

        return total_loss / self.data.num_nodes

    def fit(self, train_loader, epochs=51, loss_callback=None):
        for epoch in range(epochs):
            loss = self.train_epoch(train_loader)
            print(f'Epoch {epoch+1:02d}, Loss: {loss:.4f}')
            if loss_callback is not None:
                loss_callback(loss, epoch)

    @torch.no_grad()
    def encode(self, encoding_loader):
        self.eval()

        zs = []
        for i, (_, n_id, adjs) in enumerate(encoding_loader):
            adjs = [adj.to(self.device) for adj in adjs]
            zs.append(self.forward(self.data.x[n_id], adjs))

        out = torch.cat(zs, dim=0).cpu()
        train_mask = self.data.train_mask.cpu()        
        test_mask = self.data.test_mask.cpu()
        y = self.data.y.cpu()

        return out, y, train_mask, test_mask
