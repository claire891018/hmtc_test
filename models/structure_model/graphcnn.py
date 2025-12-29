#!/usr/bin/env python
# coding:utf-8
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np

class HierarchyGCN(nn.Module):
    def __init__(self,
                 num_nodes,
                 in_matrix,
                 out_matrix,
                 in_dim,
                 dropout=0.0,
                 device=torch.device('cpu'),
                 root=None,
                 hierarchical_label_dict=None,
                 label_trees=None):
        """
        Graph Convolutional Network variant for hierarchy structure
        original GCN paper:
                Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks.
                    arXiv preprint arXiv:1609.02907.
        :param num_nodes: int, N
        :param in_matrix: numpy.Array(N, N), input adjacent matrix for child2parent (bottom-up manner)
        :param out_matrix: numpy.Array(N, N), output adjacent matrix for parent2child (top-down manner)
        :param in_dim: int, the dimension of each node <- config.structure_encoder.node.dimension
        :param layers: int, the number of layers <- config.structure_encoder.num_layer
        :param time_step: int, the number of time steps <- config.structure_encoder.time_step
        :param dropout: Float, P value for dropout module <- configure.structure_encoder.node.dropout
        :param prob_train: Boolean, train the probability matrix if True <- config.structure_encoder.prob_train
        :param device: torch.device <- config.train.device_setting.device
        """
        super(HierarchyGCN, self).__init__()
        self.model = nn.ModuleList()
        self.model.append(
            HierarchyGCNModule(num_nodes,
                               in_matrix, out_matrix,
                               in_dim,
                               dropout,
                               device))

    def forward(self, label):
        return self.model[0](label)

class HierarchyGCNModule(nn.Module):
    def __init__(self,
                 num_nodes,
                 in_adj, out_adj,
                 in_dim, dropout, device, in_arc=True, out_arc=True,
                 self_loop=True):
        """
        module of Hierarchy-GCN
        :param num_nodes: int, N
        :param in_adj: numpy.Array(N, N), input adjacent matrix for child2parent (bottom-up manner)
        :param out_adj: numpy.Array(N, N), output adjacent matrix for parent2child (top-down manner)
        :param in_dim: int, the dimension of each node <- config.structure_encoder.node.dimension
        :param dropout: Float, P value for dropout module <- configure.structure_encoder.node.dropout
        :param prob_train: Boolean, train the probability matrix if True <- config.structure_encoder.prob_train
        :param device: torch.device <- config.train.device_setting.device
        :param in_arc: Boolean, True
        :param out_arc: Boolean, True
        :param self_loop: Boolean, True
        """

        super(HierarchyGCNModule, self).__init__()
        self.self_loop = self_loop
        self.out_arc = out_arc
        self.in_arc = in_arc
        self.device = device
        assert in_arc or out_arc
        #  bottom-up child sum
        in_prob = in_adj
        self.adj_matrix = Parameter(torch.Tensor(in_prob))
        self.edge_bias = Parameter(torch.Tensor(num_nodes, in_dim))
        self.gate_weight = Parameter(torch.Tensor(in_dim, 1))
        self.bias_gate = Parameter(torch.Tensor(num_nodes, 1))
        self.activation = nn.ReLU()
        self.origin_adj = torch.Tensor(np.where(in_adj <= 0, in_adj, 1.0)).to(device)
        # top-down: parent to child
        self.out_adj_matrix = Parameter(torch.Tensor(out_adj))
        self.out_edge_bias = Parameter(torch.Tensor(num_nodes, in_dim))
        self.out_gate_weight = Parameter(torch.Tensor(in_dim, 1))
        self.out_bias_gate = Parameter(torch.Tensor(num_nodes, 1))
        self.loop_gate = Parameter(torch.Tensor(in_dim, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        """
        initialize parameters
        """
        for param in [self.gate_weight, self.loop_gate, self.out_gate_weight]:
            nn.init.xavier_uniform_(param)
        for param in [self.edge_bias, self.out_edge_bias, self.bias_gate]:
            nn.init.zeros_(param)

    def forward(self, inputs):
        """
        :param inputs: torch.FloatTensor, (batch_size, N, in_dim)
        :return: message_ -> torch.FloatTensor (batch_size, N, in_dim)
        """
        h_ = inputs  # batch, N, in_dim
        message_ = torch.zeros_like(h_).to(self.device)  # batch, N, in_dim

        h_in_ = torch.matmul(self.origin_adj * self.adj_matrix, h_)  # batch, N, in_dim
        in_ = h_in_ + self.edge_bias
        in_ = in_
        # N,1,dim
        in_gate_ = torch.matmul(h_, self.gate_weight)
        # N, 1
        in_gate_ = in_gate_ + self.bias_gate
        in_ = in_ * F.sigmoid(in_gate_)
        in_ = self.dropout(in_)
        message_ += in_  # batch, N, in_dim

        h_output_ = torch.matmul(self.origin_adj.transpose(0, 1) * self.out_adj_matrix, h_)
        out_ = h_output_ + self.out_edge_bias
        out_gate_ = torch.matmul(h_, self.out_gate_weight)
        out_gate_ = out_gate_ + self.out_bias_gate
        out_ = out_ * F.sigmoid(out_gate_)
        out_ = self.dropout(out_)
        message_ += out_

        loop_gate = torch.matmul(h_, self.loop_gate)
        loop_ = h_ * F.sigmoid(loop_gate)
        loop_ = self.dropout(loop_)
        message_ += loop_

        return self.activation(message_)
    
class HierarchyGAT(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_matrix,
        out_matrix,
        in_dim,
        dropout,
        device,
        root=None,
        hierarchical_label_dict=None,
        label_trees=None,
        num_heads=1,
        attn_dropout=0.1,
        num_layers=1,
    ):
        super().__init__()
        
        # Convert numpy to tensor and add self-loops
        in_adj = torch.FloatTensor(in_matrix).to(device)
        out_adj = torch.FloatTensor(out_matrix).to(device)
        
        # Add self-loops (diagonal = 1)
        eye = torch.eye(num_nodes, device=device)
        in_adj = torch.clamp(in_adj + eye, max=1.0)
        out_adj = torch.clamp(out_adj + eye, max=1.0)

        self.layers = nn.ModuleList([
            HierarchyGATModule(
                in_dim=in_dim,
                out_dim=in_dim,
                in_adj=in_adj,
                out_adj=out_adj,
                num_heads=num_heads,
                dropout=attn_dropout,  # 傳遞 attn_dropout
            )
            for _ in range(num_layers)
        ])

        self.dropout = dropout

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class HierarchyGATModule(nn.Module):
    """
    Dense GAT layer for hierarchical label graph
    Input:  (B, N, D)
    Output: (B, N, D)
    """
    def __init__(self, in_dim, out_dim, in_adj, out_adj, num_heads=1, dropout=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.out_dim = out_dim
        self.dropout = dropout if dropout is not None else 0.0  # 確保不是 None

        self.fc = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.attn_src = Parameter(torch.Tensor(1, num_heads, out_dim))
        self.attn_dst = Parameter(torch.Tensor(1, num_heads, out_dim))

        # Register as buffer (won't be updated during training)
        self.register_buffer('in_adj', in_adj)
        self.register_buffer('out_adj', out_adj)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def _attention(self, h, adj):
        """
        h:   (B, N, H, D)
        adj: (N, N)
        """
        B, N, H, D = h.size()

        src_score = (h * self.attn_src).sum(dim=-1)  # (B, N, H)
        dst_score = (h * self.attn_dst).sum(dim=-1)  # (B, N, H)

        e = src_score.unsqueeze(2) + dst_score.unsqueeze(1)  # (B, N, N, H)
        e = F.leaky_relu(e, negative_slope=0.2)

        # Mask out invalid edges
        mask = adj.unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
        e = e.masked_fill(mask == 0, float("-inf"))

        alpha = F.softmax(e, dim=2)  # (B, N, N, H)
        
        # Apply dropout - 確保 self.dropout 是 float
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        # Aggregate neighbor features
        out = torch.einsum("bijn,bjhd->bihd", alpha, h)  # (B, N, H, D)
        return out

    def forward(self, x):
        """
        x: (B, N, in_dim)
        """
        B, N, _ = x.size()
        
        # Linear transformation
        h = self.fc(x).view(B, N, self.num_heads, self.out_dim)  # (B, N, H, D)

        # Bottom-up and top-down attention
        h_in = self._attention(h, self.in_adj)    # child -> parent
        h_out = self._attention(h, self.out_adj)  # parent -> child

        # Combine both directions
        h = (h_in + h_out) / 2.0  # (B, N, H, D)
        
        # Average over attention heads
        h = h.mean(dim=2)  # (B, N, out_dim)
        
        return h