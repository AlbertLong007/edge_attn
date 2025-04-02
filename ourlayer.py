import torch
import torch.nn.functional as F
import itertools
import torch.nn as nn

class BaseHOLinear(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 bias=True, 
                 nblock=2,
                 order=2,
                 set_order_weights=None, 
                 softmax_order_weights=False):
        '''
        in_channels: int, number of input channels
        out_channels: int, number of output channels
        bias: bool, whether to use bias
        nblock: int, number of blocks to split the input channels into
        order: int, order of the operator
        set_order_weights: list or None, list of order weights
        softmax_order_weights: bool, whether to apply softmax to the order weights
        '''
        super(BaseHOLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nblock = nblock
        self.order = order
        assert self.order <= self.nblock, "order should be less than or equal to nblock"

        self.dim_split = [in_channels // nblock * i for i in range(nblock)] + [in_channels]

        if set_order_weights is None:
            self.order_weights = torch.nn.Parameter(torch.Tensor([0] + [1.0] + [0] * (order-1)))
        else: 
            # fix the order weights
            assert len(set_order_weights) == order + 1, "set_order_weights should have length order + 1"
            self.register_buffer('order_weights', torch.Tensor(set_order_weights))

        self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels))
        if bias: self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else: self.bias = None

        self.softmax_order_weights = softmax_order_weights

        self.reset_parameters()

    def reset_parameters(self):
        #TODO: initialization can be modified according to the specific task
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None: torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.order == 1:
            return F.linear(x, self.weight, self.bias)
        
        ips = []
        for i in range(self.nblock):
            indices = torch.arange(self.dim_split[i], self.dim_split[i+1], dtype=torch.long).to(x.device)
            ips.append(F.linear(
                torch.index_select(x, -1, indices), 
                torch.index_select(self.weight, -1, indices)))
        # ips: [nblock, :, out_channels]
        if self.softmax_order_weights: order_weights = F.softmax(self.order_weights, dim=0)
        else: order_weights = self.order_weights
        result = torch.sum(torch.stack(ips), 0) * order_weights[1] + order_weights[0]
        for cur_order in range(2, self.order+1):
            for combination in itertools.combinations(ips, cur_order):
                ips_comb = torch.stack(combination)
                result += torch.prod(ips_comb, 0) * order_weights[cur_order]
        if self.bias is not None: return result + self.bias
        else: return result


class HODot(torch.nn.Module):
    def __init__(self, 
                nblock=2,
                order=2,
                set_order_weights=None, 
                softmax_order_weights=False):
        super(HODot, self).__init__()
        self.nblock = nblock
        self.order = order
        assert self.order <= self.nblock, "order should be less than or equal to nblock"

        if set_order_weights is None:
            self.order_weights = torch.nn.Parameter(torch.Tensor([0] + [1.0] + [0] * (order-1)))
        else: 
            # fix the order weights
            assert len(set_order_weights) == order + 1, "set_order_weights should have length order + 1"
            self.register_buffer('order_weights', torch.Tensor(set_order_weights))

        self.softmax_order_weights = softmax_order_weights

    def forward(self, x, y):
        assert x.shape == y.shape, "x and y must be of the same shape" 

        if self.order == 1:
            return torch.sum(x * y, -1)

        dim = x.shape[-1]
        dim_split = [dim // self.nblock * i for i in range(self.nblock)] + [dim]
        
        ips = []
        for i in range(self.nblock):
            indices = torch.arange(dim_split[i], dim_split[i+1], dtype=torch.long).to(x.device)
            ips.append(torch.sum(
                torch.index_select(x, -1, indices) * torch.index_select(y, -1, indices), -1))
        # ips: [nblock, :]
        if self.softmax_order_weights: order_weights = F.softmax(self.order_weights, dim=0)
        else: order_weights = self.order_weights
        result = torch.sum(torch.stack(ips), 0) * order_weights[1] + order_weights[0]
        for cur_order in range(2, self.order+1):
            for combination in itertools.combinations(ips, cur_order):
                ips_comb = torch.stack(combination)
                result += torch.prod(ips_comb, 0) * order_weights[cur_order]
        else: return result


class MyTransConvLayer(nn.Module):
    '''
    transformer with fast attention
    '''

    def __init__(self, in_channels,
                 out_channels,
                 num_heads,
                 use_weight=True):
        super().__init__()
        self.Wk = BaseHOLinear(in_channels, out_channels * num_heads, nblock=4, order=4, bias=False)
        self.Wq = BaseHOLinear(in_channels, out_channels * num_heads, nblock=4, order=4, bias=False)
        if use_weight:
            self.Wv = BaseHOLinear(in_channels, out_channels * num_heads, nblock=4, order=4, bias=False)

        self.AttDot = HODot(nblock=4, order=2, softmax_order_weights=True)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, output_attn=False):
        # feature transformation
        qs = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        ks = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            vs = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
        else:
            vs = source_input.reshape(-1, 1, self.out_channels)

        # normalize input
        qs = qs / torch.norm(qs, p=2)  # [N, H, M]
        ks = ks / torch.norm(ks, p=2)  # [L, H, M]
        N = qs.shape[0]

        # numerator
        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
        # attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
        def compute_attnum(qs, kvs):
            n, h, m = qs.shape
            _, _, d = kvs.shape
            expand_qs = qs.expand(d, n, h, m)
            expand_kvs = kvs.permute(2, 0, 1).expand(n, d, h, m).transpose(0, 1)
            # return torch.sum(expand_qs * expand_kvs, 3).permute(1, 2, 0)
            return self.AttDot(expand_qs, expand_kvs).permute(1, 2, 0)
        attention_num = compute_attnum(qs, kvs)
        attention_num += N * vs

        # denominator
        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
        # attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]
        # attention_normalizer = torch.sum(qs * ks_sum.expand_as(qs), 2)  # [N, H]
        attention_normalizer = self.AttDot(qs, ks_sum.expand_as(qs))  # [N, H]

        # attentive aggregated results
        attention_normalizer = torch.unsqueeze(
            attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        attn_output = attention_num / attention_normalizer  # [N, H, D]

        # compute attention for visualization if needed
        if output_attn:
            attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)  # [N, N]
            normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdims=True)  # [N,1]
            attention = attention / normalizer

        final_output = attn_output.mean(dim=1)

        if output_attn:
            return final_output, attention
        else:
            return final_output
        

class MyTransConvLayerV2(nn.Module):
    '''
    transformer with fast attention
    '''

    def __init__(self, in_channels,
                 out_channels,
                 num_heads,
                 use_weight=True):
        super().__init__()
        self.Wk = BaseHOLinear(in_channels, out_channels * num_heads, nblock=4, order=4, bias=False)
        self.Wq = BaseHOLinear(in_channels, out_channels * num_heads, nblock=4, order=4, bias=False)
        if use_weight:
            self.Wv = BaseHOLinear(in_channels, out_channels * num_heads, nblock=4, order=4, bias=False)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, output_attn=False):
        # feature transformation
        qs = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        ks = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            vs = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
        else:
            vs = source_input.reshape(-1, 1, self.out_channels)

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
            attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        attn_output = attention_num / attention_normalizer  # [N, H, D]

        # compute attention for visualization if needed
        if output_attn:
            attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)  # [N, N]
            normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdims=True)  # [N,1]
            attention = attention / normalizer

        final_output = attn_output.mean(dim=1)

        if output_attn:
            return final_output, attention
        else:
            return final_output