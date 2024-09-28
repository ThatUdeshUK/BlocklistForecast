import torch
import torch.nn.functional as F

from torch_geometric.nn import GraphSAGE as RawGraphSAGE


class GraphSAGE(RawGraphSAGE):
    def __init__(self, data, hidden_channels, num_layers, lr=0.01, dropout=0.5):
        super().__init__(in_channels=data.x.size(1), hidden_channels=hidden_channels,
                         num_layers=num_layers, out_channels=hidden_channels, dropout=dropout)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.data = data.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)
        
    def train_epoch(self, loader):
        self.train()

        total_loss = total_examples = 0
        for data in loader:
            data = data.to(self.device)

            self.optimizer.zero_grad()
            h = self(data.x, data.edge_index)

            h_src = h[data.edge_label_index[0]]
            h_dst = h[data.edge_label_index[1]]
            link_pred = (h_src * h_dst).sum(dim=-1)  # Inner product.

            loss = F.binary_cross_entropy_with_logits(link_pred, data.edge_label)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss) * link_pred.numel()
            total_examples += link_pred.numel()

        return total_loss / total_examples

    def fit(self, train_loader, epochs=51, loss_callback=None):
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
