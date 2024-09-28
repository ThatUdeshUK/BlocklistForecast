import torch
import torch.nn as nn

from torch_geometric.nn import DeepGraphInfomax, SAGEConv


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(SAGEConv(in_channels if i == 0 else hidden_channels, hidden_channels))
        
        self.activations = torch.nn.ModuleList()
        for i in range(num_layers):
            self.activations.append(nn.PReLU(hidden_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.activations[i](x)
        return x


def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


class DGI_SAGE(DeepGraphInfomax):
    def __init__(self, data, hidden_channels, num_layers, lr=0.001):
        super().__init__(
            hidden_channels,
            encoder=Encoder(data.x.size(1), hidden_channels, num_layers),
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=corruption
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.data = data.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.to(self.device)
        
    def train_epoch(self, train_loader):        
        self.train()

        total_loss = total_examples = 0
        for data in train_loader:
            data = data.to(self.device)
            batch_size = data.batch_size

            self.optimizer.zero_grad()
            pos_z, neg_z, summary = self.forward(data.x, data.edge_index)
            loss = self.loss(pos_z[:batch_size], neg_z[:batch_size], summary)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * batch_size
            total_examples += batch_size

        return total_loss / total_examples

    def fit(self, train_loader, epochs=31, loss_callback=None):
        for epoch in range(epochs):
            loss = self.train_epoch(train_loader)
            print(f'Epoch {epoch+1:02d}, Loss: {loss:.4f}')
            if loss_callback is not None:
                loss_callback(loss, epoch)

    @torch.no_grad()
    def encode(self, encoding_loader):        
        self.eval()

        out = []
        train_mask = []
        test_mask = []
        y = []
        for data in encoding_loader:
            data = data.to(self.device)
            batch_size = data.batch_size
            
            out.append(self.forward(data.x, data.edge_index)[0][:batch_size])
            
            train_mask.append(data.train_mask[:batch_size])
            test_mask.append(data.test_mask[:batch_size])
            
            y.append(data.y[:batch_size])
            
        out = torch.cat(out, dim=0).cpu()
        train_mask = torch.cat(train_mask, dim=0).cpu()        
        test_mask = torch.cat(test_mask, dim=0).cpu()
        y = torch.cat(y, dim=0).cpu()
        
        return out, y, train_mask, test_mask
