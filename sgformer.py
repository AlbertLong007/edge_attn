import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul


def full_attention_conv(qs, ks, vs, output_attn=False):
    """
    qs: query tensor [N, H, M]
    ks: key tensor [L, H, M]
    vs: value tensor [L, H, D]

    return output [N, H, D]
    """
    # normalize input
    qs = qs / torch.norm(qs, p=2)  # [N, H, M]
    ks = ks / torch.norm(ks, p=2)  # [L, H, M]
    N = qs.shape[0]

    # numerator
    kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
    attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
    attention_num += N * vs

    # denominator
    all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
    attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

    # attentive aggregated results
    attention_normalizer = torch.unsqueeze(
        attention_normalizer, len(attention_normalizer.shape)
    )  # [N, H, 1]
    attention_normalizer += torch.ones_like(attention_normalizer) * N
    attn_output = attention_num / attention_normalizer  # [N, H, D]

    # compute attention for visualization if needed
    if output_attn:
        attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)  # [N, N]
        normalizer = attention_normalizer.squeeze(dim=-1).mean(
            dim=-1, keepdims=True
        )  # [N,1]
        attention = attention / normalizer

    if output_attn:
        return attn_output, attention
    else:
        return attn_output


class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_weight=True, use_init=False):
        super(GraphConvLayer, self).__init__()

        self.use_init = use_init
        self.use_weight = use_weight
        if self.use_init:
            in_channels_ = 2 * in_channels
        else:
            in_channels_ = in_channels
        self.W = nn.Linear(in_channels_, out_channels)

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, x, edge_index, x0):
        N = x.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm_in = (1.0 / d[col]).sqrt()
        d_norm_out = (1.0 / d[row]).sqrt()
        value = torch.ones_like(row) * d_norm_in * d_norm_out
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        x = matmul(adj, x)  # [N, D]

        if self.use_init:
            x = torch.cat([x, x0], 1)
            x = self.W(x)
        elif self.use_weight:
            x = self.W(x)

        return x


class GraphConv(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=2,
        dropout=0.5,
        use_bn=True,
        use_residual=True,
        use_weight=True,
        use_init=False,
        use_act=True,
    ):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers):
            self.convs.append(
                GraphConvLayer(hidden_channels, hidden_channels, use_weight, use_init)
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index):
        layer_ = []

        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, layer_[0])
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x + layer_[-1]
        return x


class EdgesConvLayer(nn.Module):
    def __init__(self, num_node_type, num_edge_type, in_channels, out_channels, n_heads, dropout=0.2, use_norm=False):
        super().__init__()
        self.num_node_type = num_node_type
        self.num_edge_type = num_edge_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.d_k = out_channels // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        
        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm
        for _ in range(num_node_type):
            self.k_linears.append(nn.Linear(in_channels, out_channels))
            self.q_linears.append(nn.Linear(in_channels, out_channels))
            self.v_linears.append(nn.Linear(in_channels, out_channels))
            self.a_linears.append(nn.Linear(out_channels, out_channels))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_channels))

        self.relation_pri = nn.Parameter(torch.ones(num_edge_type, n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(num_edge_type, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(num_edge_type, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(num_node_type))
        self.drop = nn.Dropout(dropout)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, x, type_id, edge_index, edge_attr):
        relation_att = self.relation_att[edge_attr.view(-1)]  #[E, H]
        relation_pri = self.relation_pri[edge_attr.view(-1)]  #[E, H, D]
        relation_msg = self.relation_msg[edge_attr.view(-1)]  #[E, H, D]

        src, dst = edge_index[0, :], edge_index[1, :]
        feat_src = torch.index_select(x, dim=0, index=src)
        feat_dst = torch.index_select(x, dim=0, index=dst)

        ##TODO
        k_src = torch.zeros(feat_src.size(0), self.out_channels)
        q_dst = torch.zeros(feat_dst.size(0), self.out_channels)
        v_src = torch.zeros(feat_src.size(0), self.out_channels)
        for t in range(len(self.num_node_type)):
            mask = (edge_attr == t).view(-1)
            if mask.any():
                k_src[mask] = self.k_linears[t](feat_src[mask])
                q_dst[mask] = self.q_linears[t](feat_dst[mask])
                v_src[mask] = self.v_linears[t](feat_src[mask])

        k_src = k_src.view(-1, self.n_heads, self.d_k)
        q_dst = q_dst.view(-1, self.n_heads, self.d_k)
        v_src = v_src.view(-1, self.n_heads, self.d_k)
        
        k_trans = torch.einsum('ehd,ehdk->ehk', feat_src, relation_att)
        att = (feat_dst * k_trans).sum(dim=-1) * relation_pri / self.sqrt_dk  #[E, H]
        att_softmax = torch_scatter.composite.scatter_softmax(att, dst, dim=0)  #[E, H]
        v_trans = torch.einsum('ehd,ehdk->ehk', feat_src, relation_msg)

        weighted = att_softmax.unsqueeze(-1) * v_trans  # [E, H, D]
        weighted = weighted.view(weighted.size(0), -1)  # [E, Hidden]

        x_0 = torch.zeros_like(x)
        x_0 = torch_scatter.scatter_mean(weighted, dst, dim=0, out=x_0)  # [N, Hidden]

        skip = self.skip[type_id]
        alpha = torch.sigmoid(skip).unsqueeze(-1)

        ##TODO
        for t in range(len(self.num_node_type)):  # 遍历所有类型
            mask = (edge_attr == t).view(-1)
            if mask.any():
                x_0[mask] = self.a_linears[t](x_0[mask])

        # residual
        x_0 = x_0 * alpha + x * (1-alpha)

        if self.use_norm:
            gamma = torch.stack([norm.weight for norm in self.norms])  # [num_types, 16]
            beta = torch.stack([norm.bias for norm in self.norms])  # [num_types, 16]
            
            # 手动计算LayerNorm
            eps = self.norms[0].eps  # 假设所有LayerNorm的eps相同
            mean = x_0.mean(dim=1, keepdim=True)
            var = x_0.var(dim=1, keepdim=True, unbiased=False)
            x_normed = (x_0 - mean) / torch.sqrt(var + eps)
            
            # 根据type_id选择对应的gamma和beta
            selected_gamma = gamma[type_id]  # [N, 16]
            selected_beta = beta[type_id]    # [N, 16]
            
            # 应用缩放、偏移和Dropout
            x_0 = self.drop(x_normed * selected_gamma + selected_beta)
        else:
            x_0 = self.drop(x_0)
        
        # 返回结果
        return x_0


class EdgesConv(nn.Module):
    def __init__(
            self,
            num_node_type,
            num_edge_type,
            in_channels,
            hidden_channels,
            num_layers,
            num_heads,
            dropout,
            use_norm=True
        ):
        super().__init__()
        self.ecs = nn.ModuleList()
        self.num_layers = num_layers
        
        for _ in range(num_layers):
            self.ecs.append(
                EdgesConvLayer(
                    num_node_type,
                    num_edge_type,
                    in_channels,
                    hidden_channels,
                    num_heads,
                    dropout,
                    use_norm=use_norm,
                )
            )
    
    def reset_parameters(self):
        for ec in self.ecs:
            ec.reset_parameters()

    def forward(self, x, edge_index, edge_attr):        
        for i in range(self.num_layers):
            x = self.ecs[i](x, edge_index, edge_attr)
        
        return x


class TransConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, edge_index=None, output_attn=False):
        # feature transformation
        query = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            value = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
        else:
            value = source_input.reshape(-1, 1, self.out_channels)

        # compute full attentive aggregation
        if output_attn:
            attention_output, attn = full_attention_conv(
                query, key, value, output_attn
            )  # [N, H, D]
        else:
            attention_output = full_attention_conv(query, key, value)  # [N, H, D]

        final_output = attention_output.mean(dim=1)

        if output_attn:
            return final_output, attn
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=2,
        num_heads=1,
        alpha=0.5,
        dropout=0.5,
        use_bn=True,
        use_residual=True,
        use_weight=True,
        use_act=True,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                TransConvLayer(
                    hidden_channels,
                    hidden_channels,
                    num_heads=num_heads,
                    use_weight=use_weight,
                )
            )
            self.bns.append(nn.LayerNorm(hidden_channels))

        # self.fcs.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.alpha = alpha
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index=None):
        layer_ = []

        # input MLP layer
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # store as residual link
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with full attention aggregation
            x = conv(x, x, edge_index)
            if self.use_residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.use_residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]


class SGFormer(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        trans_num_layers=1,
        trans_num_heads=1,
        trans_dropout=0.5,
        gnn_num_layers=1,
        gnn_dropout=0.5,
        gnn_use_weight=True,
        gnn_use_init=False,
        gnn_use_bn=True,
        gnn_use_residual=True,
        gnn_use_act=True,
        alpha=0.5,
        trans_use_bn=True,
        trans_use_residual=True,
        trans_use_weight=True,
        trans_use_act=True,
        use_graph=True,
        graph_weight=0.8,
        aggregate="add",
    ):
        super().__init__()
        self.trans_conv = TransConv(
            in_channels,
            hidden_channels,
            trans_num_layers,
            trans_num_heads,
            alpha,
            trans_dropout,
            trans_use_bn,
            trans_use_residual,
            trans_use_weight,
            trans_use_act,
        )
        self.graph_conv = GraphConv(
            in_channels,
            hidden_channels,
            gnn_num_layers,
            gnn_dropout,
            gnn_use_bn,
            gnn_use_residual,
            gnn_use_weight,
            gnn_use_init,
            gnn_use_act,
        )
        self.use_graph = use_graph
        self.graph_weight = graph_weight

        self.aggregate = aggregate

        if aggregate == "add":
            self.fc = nn.Linear(hidden_channels, out_channels)
        elif aggregate == "cat":
            self.fc = nn.Linear(2 * hidden_channels, out_channels)
        else:
            raise ValueError(f"Invalid aggregate type:{aggregate}")

        self.params1 = list(self.trans_conv.parameters())
        self.params2 = (
            list(self.graph_conv.parameters()) if self.graph_conv is not None else []
        )
        self.params2.extend(list(self.fc.parameters()))

    def forward(self, x, edge_index):
        x1 = self.trans_conv(x)
        #x1 和其他embedding通过sum / fc 聚合得到新的x1
        if self.use_graph:
            x2 = self.graph_conv(x, edge_index)
            if self.aggregate == "add":
                x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
            else:
                x = torch.cat((x1, x2), dim=1)
        else:
            x = x1
        x = self.fc(x)
        return x

    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x)  # [layer num, N, N]

        return attns

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.graph_conv.reset_parameters()
