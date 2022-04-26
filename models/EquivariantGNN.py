import torch
from torch.nn import Linear, Module
from models.EquivariantLayer import EquivariantLayer
from torch_geometric.nn import global_mean_pool


class EquivariantGNN(Module):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1, pos_aggr='mean', orthogonal_interactions=False):
        """Message Passing Neural Network model for graph property prediction

        This model uses both node features and coordinates as inputs, and
        is invariant to 3D rotations and translations (the constituent MPNN layers
        are equivariant to 3D rotations and translations).

        Args:
            num_layers: (int) - number of message passing layers `L`
            emb_dim: (int) - hidden dimension `d`
            in_dim: (int) - initial node feature dimension `d_n`
            edge_dim: (int) - edge feature dimension `d_e`
            out_dim: (int) - output dimension (fixed to 1)
        """
        super().__init__()

        # Linear projection for initial node features
        # dim: d_n -> d
        self.lin_in = Linear(in_dim, emb_dim)

        # Stack of MPNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(EquivariantLayer(
                emb_dim, edge_dim, aggr='add', interaction_aggr=pos_aggr, orthogonal_interactions=orthogonal_interactions))

        # Linear prediction head
        # dim: d -> out_dim
        self.lin_pred = Linear(emb_dim, out_dim)

    def forward(self, data):
        """
        Args:
            data: (PyG.Data) - batch of PyG graphs

        Returns: 
            out: (batch_size, out_dim) - prediction for each graph
        """
        h = self.lin_in(data.x)  # (n, d_n) -> (n, d)
        pos = data.pos

        for conv in self.convs:
            # Message passing layer
            h_update, pos_update = conv(
                h, pos, data.edge_index, data.edge_attr)

            # Update node features
            h = h + h_update  # (n, d) -> (n, d)
            # Note that we add a residual connection after each MPNN layer

            # Update node coordinates
            pos = pos_update  # (n, 3) -> (n, 3)

        out = self.lin_pred(h)  # (batch_size, d) -> (batch_size, 1)

        return out

    def reset_parameters(self):
        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
            elif isinstance(module, torch.nn.ModuleList):
                for layer in module.children():
                    layer.reset_parameters()
