# An implementation of "Unsupervised Attributed Multiplex Network
# Embedding" (DMGI) for unsupervised learning on  heterogeneous graphs:
# * Paper: <https://arxiv.org/abs/1911.06750> (AAAI 2020)


# Diff with ori DMGI
# * No highlighting the self edge by 3x
# * Norm features - fixed (done at preproc)
# * Sigmoid after disc - fixed
# * Custom CE loss vs BCELogitLoss - fixed

# Not checked
# * Diffs in GCN layers


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from copy import deepcopy


class DMGI(torch.nn.Module):
    def __init__(self, data, hidden_channels, alpha=0.001, lr=0.0005, weight_decay=0.0001, head_node='domain_node'):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = data.to(self.device)
        self.head_node = head_node
        
        num_relations = len(data.edge_index_dict)
        self.convs = torch.nn.ModuleList([
            GCNConv(data[head_node].x.size(1), hidden_channels) for _ in range(num_relations) # Matched (Check this - gcn)
        ])
        self.M = torch.nn.Bilinear(hidden_channels, hidden_channels, 1)
        self.Z = torch.nn.Parameter(torch.Tensor(data[head_node].x.size(0), hidden_channels)) # Matched
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.reset_parameters() # Matched
        
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay) # Matching

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.M.weight)
        self.M.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.Z)

    def forward(self, x, edge_indices):
        pos_hs, neg_hs, summaries = [], [], []
        for conv, edge_index in zip(self.convs, edge_indices):
            pos_h = F.dropout(x, p=0.5, training=self.training)
            pos_h = conv(pos_h, edge_index).relu()
            pos_hs.append(pos_h) # Pos matched

            neg_h = F.dropout(x, p=0.5, training=self.training)
            neg_h = neg_h[torch.randperm(neg_h.size(0), device=neg_h.device)] # SHuffle matched
            neg_h = conv(neg_h, edge_index).relu() # Neg matched
            neg_hs.append(neg_h)

            summaries.append(pos_h.mean(dim=0, keepdim=True).sigmoid()) # Readout function/Summary (Matching)
            # Mismatch (Sigmoid) -fixed

        return pos_hs, neg_hs, summaries

    def loss(self, pos_hs, neg_hs, summaries):
        loss = 0.
        for pos_h, neg_h, s in zip(pos_hs, neg_hs, summaries):
            s = s.expand_as(pos_h)
            
            # Matched disc. (apply biliner with self.M(h, s))
            # Miss-matched with add. EPS, and sigmoid after disc - fixed
            
            pos_disc = self.M(pos_h, s)
            neg_disc = self.M(neg_h, s)
            
            logit = torch.cat((pos_disc, neg_disc))
            lbl = torch.cat((torch.ones((pos_h.size(0),1)), torch.zeros((neg_h.size(0),1)))).to(self.device)
            
            # Mismatched (do manually CE) - change this to BCELogitLoss - fixed
            loss += self.bce_loss(logit, lbl)

        pos_mean = torch.stack(pos_hs, dim=0).mean(dim=0) # Matched
        neg_mean = torch.stack(neg_hs, dim=0).mean(dim=0) # Matched

        pos_reg_loss = (self.Z - pos_mean).pow(2).sum() # Matched
        neg_reg_loss = (self.Z - neg_mean).pow(2).sum() # Matched
        loss += 0.001 * (pos_reg_loss - neg_reg_loss) # Matched

        return loss

    def train_epoch(self):
        self.train()
        self.optimizer.zero_grad()
        x = self.data[self.head_node].x
        edge_indices = self.data.edge_index_dict.values()
        pos_hs, neg_hs, summaries = self.forward(x, edge_indices)
        loss = self.loss(pos_hs, neg_hs, summaries)
        loss.backward()
        self.optimizer.step()
        return float(loss)
    
    def fit(self, epochs=31, patience=20, loss_callback=None):
        cnt_wait = 0
        best = 1e9
        for epoch in range(epochs):
            loss = self.train_epoch()
            print(f'Epoch {epoch+1:02d}, Loss: {loss:.4f}')
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

        self.load_state_dict(saved)
                
    @torch.no_grad()
    def encode(self):        
        self.eval()

        out = self.Z.cpu()
        train_mask = self.data[self.head_node].train_mask.cpu()
        test_mask = self.data[self.head_node].test_mask.cpu()
        y = self.data[self.head_node].y.cpu()

        return out, y, train_mask, test_mask
