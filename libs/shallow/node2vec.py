import torch
from torch_geometric.nn import Node2Vec as RawNode2Vec


class Node2Vec(RawNode2Vec):
    def __init__(self, data, hidden_channels=128, walk_length=20, walks_per_node=10, p=1, q=1, batch_size=256,
                 lr=0.01):
        super().__init__(data.edge_index, embedding_dim=hidden_channels, walk_length=walk_length,
                     context_size=10, walks_per_node=walks_per_node,
                     num_negative_samples=1, p=p, q=q, sparse=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = data.to(self.device)
        self.batch_size = batch_size
        self.to(self.device)
        
        self.optimizer = torch.optim.SparseAdam(list(self.parameters()), lr=lr)

    def train_epoch(self, loader):
        self.train()
        
        total_loss = 0
        for pos_rw, neg_rw in loader:
            self.optimizer.zero_grad()
            loss = self.loss(pos_rw.to(self.device), neg_rw.to(self.device))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def fit(self, epochs=100, loss_callback=None):
        loader = self.loader(batch_size=self.batch_size, shuffle=True, num_workers=12)

        for epoch in range(epochs):
            loss = self.train_epoch(loader)
            print(f'Epoch: {epoch + 1:03d}, Loss: {loss:.4f}')
            if loss_callback is not None:
                loss_callback(loss, epoch)
            
    @torch.no_grad()
    def encode(self):        
        self.eval()
        
        out = self().cpu()
        y = self.data.y.cpu()
        train_mask = self.data.train_mask.cpu()
        test_mask = self.data.test_mask.cpu()
        
        return out, y, train_mask, test_mask
