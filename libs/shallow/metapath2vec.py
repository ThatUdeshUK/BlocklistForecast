
import torch
from torch_geometric.nn import MetaPath2Vec as RawMetaPath2Vec


class MetaPath2Vec(RawMetaPath2Vec):
    def __init__(self, data, metapaths, hidden_channels=128, walk_length=20, walks_per_node=10, batch_size=256,
                 lr=0.01):
        super().__init__(data.edge_index_dict, embedding_dim=hidden_channels, metapath=metapaths, 
                         walk_length=walk_length, context_size=10, walks_per_node=walks_per_node,
                         num_negative_samples=1, sparse=True)
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
            
    def fit(self, epochs=10, loss_callback=None):
        loader = self.loader(batch_size=self.batch_size, shuffle=True, num_workers=12)

        for epoch in range(epochs):
            loss = self.train_epoch(loader)
            print(f'Epoch: {epoch + 1:03d}, Loss: {loss:.4f}')
            if loss_callback is not None:
                loss_callback(loss, epoch)
            
    @torch.no_grad()
    def encode(self):        
        self.eval()
        
        out = self('domain_node').cpu()
        y = self.data['domain_node'].y.cpu()
        train_mask = self.data['domain_node'].train_mask.cpu()
        test_mask = self.data['domain_node'].test_mask.cpu()
        
        return out, y, train_mask, test_mask


