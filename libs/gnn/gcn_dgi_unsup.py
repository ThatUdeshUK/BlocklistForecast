import torch
import torch.nn as nn

from torch_geometric.nn import DeepGraphInfomax, GCNConv


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            GCNConv(in_channels, hidden_channels),
            GCNConv(hidden_channels, hidden_channels)
        ])

        self.activations = torch.nn.ModuleList([
            nn.PReLU(hidden_channels),
            nn.PReLU(hidden_channels)
        ])

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.activations[i](x)
        return x


def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


class DGI_GCN(DeepGraphInfomax):
    def __init__(self, data, in_channels, hidden_channels, lr=0.0001):
        super().__init__(
            hidden_channels,
            encoder=Encoder(in_channels, hidden_channels),
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=corruption
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.data = data.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.to(self.device)
        
    def train_epoch(self):        
        x, edge_index = self.data.x, self.data.edge_index
        self.train()
            
        self.optimizer.zero_grad()
        pos_z, neg_z, summary = self.forward(x, edge_index)
        loss = self.loss(pos_z, neg_z, summary)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def fit(self, epochs=31, loss_callback=None):
        for epoch in range(epochs):
            loss = self.train_epoch()
            print(f'Epoch {epoch+1:02d}, Loss: {loss:.4f}')
            if loss_callback is not None:
                loss_callback(loss, epoch)

    @torch.no_grad()
    def encode(self):
        self.eval()

        out = self.forward(self.data.x, self.data.edge_index)[0].cpu()
        train_mask = self.data.train_mask.cpu()
        test_mask = self.data.test_mask.cpu()
        y = self.data.y.cpu()

        return out, y, train_mask, test_mask
