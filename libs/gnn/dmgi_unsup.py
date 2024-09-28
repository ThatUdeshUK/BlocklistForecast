# An implementation of "Unsupervised Attributed Multiplex Network
# Embedding" (DMGI) for unsupervised learning on  heterogeneous graphs:
# * Paper: <https://arxiv.org/abs/1911.06750> (AAAI 2020)


import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from copy import deepcopy
import time


class DMGI(torch.nn.Module):
    def __init__(self, data, hidden_channels, alpha=0.001, lr=0.0005, weight_decay=0.0001, head_node='domain_node'):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = data.to(self.device)
        self.alpha = alpha
        self.head_node = head_node

        num_relations = len(data.edge_index_dict)
        self.convs = torch.nn.ModuleList([
            SAGEConv(data[head_node].x.size(1), hidden_channels) for _ in range(num_relations)
        ])
        self.M = torch.nn.Bilinear(hidden_channels, hidden_channels, 1)
        self.Z = torch.nn.Parameter(torch.Tensor(self.data[head_node].x.size(0), hidden_channels))
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.reset_parameters()
        
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.M.weight)
        self.M.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.Z)

    def forward(self, x, edge_indices):
        pos_hs, neg_hs = [], []
        for conv, edge_index in zip(self.convs, edge_indices):
            pos_h = F.dropout(x, p=0.5, training=self.training)
            pos_h = conv(pos_h, edge_index).relu()
            pos_hs.append(pos_h)

            neg_h = F.dropout(x, p=0.5, training=self.training)
            neg_h = neg_h[torch.randperm(neg_h.size(0), device=neg_h.device)]
            neg_h = conv(neg_h, edge_index).relu()
            neg_hs.append(neg_h)

        return pos_hs, neg_hs
    
    def disc(self, h, s):
        return self.M(h, s)  # Eq. 5 (without sigmoid which is applied with BCELogitLoss)

    def loss(self, n_id, pos_hs, neg_hs, summaries):
        loss = 0.
        for pos_h, neg_h, s in zip(pos_hs, neg_hs, summaries):
            s = s.expand_as(pos_h)
                      
            pos_disc = self.M(pos_h, s)
            neg_disc = self.M(neg_h, s)
            
            logit = torch.cat((pos_disc, neg_disc))
            lbl = torch.cat((torch.ones((pos_h.size(0), 1)), torch.zeros((neg_h.size(0), 1)))).to(self.device)
            
            loss += self.bce_loss(logit, lbl)

        pos_mean = torch.stack(pos_hs, dim=0).mean(dim=0)
        neg_mean = torch.stack(neg_hs, dim=0).mean(dim=0)

        pos_reg_loss = (self.Z[n_id] - pos_mean).pow(2).sum()
        neg_reg_loss = (self.Z[n_id] - neg_mean).pow(2).sum()
        
        loss += self.alpha * (pos_reg_loss - neg_reg_loss)

        return loss
    
    def train_epoch(self, train_loader):
        self.train()
                
        total_loss = total_examples = 0
        for data in train_loader:
            data = data.to(self.device)
            batch_size = data[self.head_node].batch_size
            
            self.optimizer.zero_grad()
            edge_indices = data.edge_index_dict.values()
            pos_hs, neg_hs = self.forward(data[self.head_node].x, edge_indices)
            
            summaries = [pos_h[:batch_size].mean(dim=0, keepdim=True).sigmoid() for pos_h in pos_hs]
            
            loss = self.loss(data[self.head_node].n_id[:batch_size], [x[:batch_size] for x in pos_hs], 
                             [x[:batch_size] for x in neg_hs], summaries)
            loss.backward()
            self.optimizer.step()
            
            total_loss += float(loss) * batch_size
            total_examples += batch_size
        
        return total_loss / total_examples
    
    def fit(self, train_loader, epochs=31, patience=30, loss_callback=None, verbose=True):
        cnt_wait = 0
        best = 1e9
        for epoch in range(epochs):        
            st = time.time()
            loss = self.train_epoch(train_loader)
            if verbose:
                et = time.time()
                print(f'Epoch {epoch+1:02d}, Loss: {loss:.4f}, Time: {int(et - st)}')
            if loss_callback is not None:
                loss_callback(loss, epoch)
                
            if loss < best:
                best = loss
                cnt_wait = 0
                saved = deepcopy(self.state_dict())
            else:
                cnt_wait += 1

            if cnt_wait == patience:
                break
                
        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(saved)
    
    @torch.no_grad()
    def encode(self):        
        self.eval()

        out = self.Z.cpu()
        train_mask = self.data[self.head_node].train_mask.cpu()
        test_mask = self.data[self.head_node].test_mask.cpu()
        y = self.data[self.head_node].y.cpu()

        return out, y, train_mask, test_mask
