from torch_geometric.typing import Adj, OptTensor
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import SGConv


class CustomSGConv(SGConv):

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        x = x.to_dense()
        return super().forward(x, edge_index, edge_weight)

# Use same forward() call signature as the GNNs


class SGC(CustomSGConv):
    def forward(self, data: Data):
        return super().forward(data.x, data.edge_index)
