import torch
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter


class EquivariantLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=4, aggr='add', interaction_aggr='mean', orthogonal_interactions=False):
        """Message Passing Neural Network Layer

        This layer is equivariant to 3D rotations and translations.

        Args:
            emb_dim: (int) - hidden dimension `d`
            edge_dim: (int) - edge feature dimension `d_e`
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
            interaction_aggr: (str) - aggregation function for position interactions (sum/mean/max)
        """
        # Set the aggregation function
        super().__init__(aggr=aggr)
        self.interaction_aggr = interaction_aggr
        self.orthogonal_interactions = orthogonal_interactions

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        self.mlp_msg = Sequential(
            Linear(2*emb_dim + edge_dim + 1,
                   emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
        )
        self.mlp_upd = Sequential(
            Linear(2*emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
        )
        self.mlp_position_upd = Sequential(
            Linear(2*emb_dim + edge_dim + 1,
                   emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, 1), BatchNorm1d(1), ReLU()
        )
        self.mlp_position_upd_normal = Sequential(
            Linear(2*emb_dim + edge_dim + 1,
                   emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, 1), BatchNorm1d(1), ReLU()
        )

    def forward(self, h, pos, edge_index, edge_attr):
        """
        The forward pass updates node features `h` via one round of message passing.

        Args:
            h: (n, d) - initial node features
            pos: (n, 3) - initial node coordinates
            edge_index: (e, 2) - pairs of edges (i, j)
            edge_attr: (e, d_e) - edge features

        Returns:
            out: [(n, d),(n,3)] - updated node features
        """
        out = self.propagate(
            edge_index, h=h, edge_attr=edge_attr, pos=pos, n_nodes=h.shape[0])
        return out

    def message(self, h_i, h_j, edge_attr, pos_i, pos_j):
        """The `message()` function constructs messages from source nodes j 
        to destination nodes i for each edge (i, j) in `edge_index`.

        Args:
            h_i: (e, d) - destination node features
            h_j: (e, d) - source node features
            edge_attr: (e, d_e) - edge features
            pos_i: (e, 3) - destination node coordinates
            pos_j: (e, 3) - source node coordinates

        Returns:
            msg: ((e, d), (e, 3)) - messages `m_ij` passed through MLP `\psi`, 
                                    "interatomic force" vector for each edge
        """
        dist = torch.norm(pos_i - pos_j, dim=1).reshape(-1, 1)
        if self.edge_dim:
            msg = torch.cat([h_i, h_j, edge_attr, dist], dim=-1)
        else:
            msg = torch.cat([h_i, h_j, dist], dim=-1)

        # Compute the strength of position interactions
        force_scalar = self.mlp_position_upd(msg)
        if self.orthogonal_interactions:
            force_scalar_normal = self.mlp_position_upd_normal(msg)
        # Get direction of interactions
        force_direction = pos_j - pos_i
        # Normalize direction vectors
        force_direction = (force_direction / dist)
        if self.orthogonal_interactions:
            orthogonal_direction = torch.matmul(
                force_direction, torch.tensor([[0., -1.], [1., 0.]]))
        # Get interatomic force vectors
        if self.orthogonal_interactions:
            force_vector = force_scalar * force_direction + \
                force_scalar_normal * orthogonal_direction
        else:
            force_vector = force_scalar * force_direction
        return self.mlp_msg(msg), force_vector

    def aggregate(self, inputs, index, n_nodes):
        """The `aggregate` function aggregates the messages from neighboring nodes,
        according to the chosen aggregation function ('sum' by default).

        Args:
            inputs: ((e, d) (e, d)) - tuple of messages `m_ij` from destination to source nodes
            index: (e, 1) - list of source nodes for each edge/message in `input`

        Returns:
            aggr_out: ((n, d), (n, 3)) - aggregated messages and position updates
        """
        # print(n_nodes, index)
        aggr_h = scatter(inputs[0], index, dim=self.node_dim,
                         reduce=self.aggr, dim_size=n_nodes)
        aggr_x = scatter(inputs[1], index, dim=self.node_dim,
                         reduce=self.interaction_aggr, dim_size=n_nodes)
        return aggr_h, aggr_x

    def update(self, aggr_out, h, pos):
        """The `update()` function computes the final node features by combining the 
        aggregated messages with the initial node features.

        Args:
            aggr_out: ((n, d), (n, 3)) - aggregated messages and position updates
            h: (n, d) - initial node features

        Returns:
            upd_out: ((n, d), (n, 3)) - updated node features passed through MLP `\phi`,
                                        updated node coordinates
        """
        aggr_h, aggr_x = aggr_out
        # print(h.shape, aggr_h.shape)
        h_upd_out = torch.cat([h, aggr_h], dim=-1)
        x_upd_out = pos + aggr_x
        return self.mlp_upd(h_upd_out), x_upd_out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')

    def reset_parameters(self):
        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
            elif isinstance(module, torch.nn.Sequential):
                for submodule in module:
                    if hasattr(submodule, 'reset_parameters'):
                        submodule.reset_parameters()
