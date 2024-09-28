import torch
import torch.nn as nn

from typing import Union, Tuple
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size
from torch import Tensor

import torch.nn.functional as F
from torch_geometric.nn import DeepGraphInfomax, HeteroConv
from torch_geometric.nn import HeteroConv
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear


class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j
    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)


class Encoder(nn.Module):
    def __init__(self, hidden_channels, num_layers, edge_types):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels) for edge_type in edge_types
            }, aggr='sum'))

        self.activations = torch.nn.ModuleList()
        for i in range(num_layers):
            self.activations.append(nn.PReLU(hidden_channels))

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        if edge_weight_dict is None:
            edge_weight_dict = {k: None for k in edge_index_dict.keys()}
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict, edge_weight_dict)
#             x_dict = {k: F.relu(x) for k, x in x_dict.items()}
            x_dict = {k: self.activations[i](x) for k, x in x_dict.items()}
        return x_dict['domain_node']


def corruption(x_dict, edge_index_dict, edge_weight_dict=None):
    return {k: x[torch.randperm(x.size(0))] if k == 'domain_node' else x for k, x in x_dict.items()}, edge_index_dict


class DGI_HinSAGE(DeepGraphInfomax):
    def __init__(self, data, hidden_channels, num_layers, lr=0.0001):
        super().__init__(
            hidden_channels,
            encoder=Encoder(hidden_channels, num_layers, data.metadata()[1]),
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
            batch_size = data['domain_node'].batch_size

            self.optimizer.zero_grad()
            pos_z, neg_z, summary = self.forward(data.x_dict, data.edge_index_dict)
            loss = self.loss(pos_z[:batch_size], neg_z[:batch_size], summary)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * batch_size
            total_examples += batch_size

        return total_loss / total_examples

    def fit(self, train_loader, epochs=31, loss_callback=None):
        with torch.no_grad():  # Initialize lazy modules.
            batch = next(iter(train_loader))
            self.forward(batch.x_dict, batch.edge_index_dict)
        
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
            batch_size = data['domain_node'].batch_size
            
            out.append(self.forward(data.x_dict, data.edge_index_dict)[0][:batch_size])
            
            train_mask.append(data['domain_node'].train_mask[:batch_size])
            test_mask.append(data['domain_node'].test_mask[:batch_size])
            
            y.append(data['domain_node'].y[:batch_size])
            
        out = torch.cat(out, dim=0).cpu()
        train_mask = torch.cat(train_mask, dim=0).cpu()        
        test_mask = torch.cat(test_mask, dim=0).cpu()
        y = torch.cat(y, dim=0).cpu()
        
        return out, y, train_mask, test_mask
