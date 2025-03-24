import scipy.io
import urllib.request
import dgl
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from scipy import io as sio
from dgl.data.utils import download, get_download_dir, _get_dgl_url
import os
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='acm', choices=['acm', 'dblp', 'synthetic'],
                    help='Dataset to use: acm, dblp, or synthetic')
parser.add_argument('--local_path', type=str, default=None, help='Local path to ACM.mat file')
args = parser.parse_args()

# 1. 加载ACM数据集
def load_acm_dataset(local_path=None):
    if local_path and os.path.exists(local_path):
        data_path = local_path
        print(f"Loading ACM dataset from local path: {local_path}")
    else:
        url = 'dataset/ACM.mat'
        data_path = get_download_dir() + '/ACM.mat'
        download(_get_dgl_url(url), path=data_path)
        print(f"Downloaded ACM dataset to: {data_path}")
    
    data = sio.loadmat(data_path)
    
    # 创建DGL图
    G = dgl.heterograph({
        ('paper', 'written-by', 'author'): data['PvsA'].nonzero(),
        ('author', 'writing', 'paper'): data['PvsA'].transpose().nonzero(),
        ('paper', 'citing', 'paper'): data['PvsP'].nonzero(),
        ('paper', 'cited', 'paper'): data['PvsP'].transpose().nonzero(),
        ('paper', 'is-about', 'subject'): data['PvsL'].nonzero(),
        ('subject', 'has', 'paper'): data['PvsL'].transpose().nonzero(),
    })
    
    pvc = data['PvsC'].tocsr()
    p_selected = pvc.tocoo()
    # 生成标签
    labels = pvc.indices
    labels = torch.tensor(labels).long()
    
    # 生成训练/验证/测试集
    pid = p_selected.row
    shuffle = np.random.permutation(pid)
    train_idx = torch.tensor(shuffle[0:800]).long()
    val_idx = torch.tensor(shuffle[800:900]).long()
    test_idx = torch.tensor(shuffle[900:]).long()
    
    return G, labels, train_idx, val_idx, test_idx

# 2. 加载DBLP数据集（从DGL数据集）
def load_dblp_dataset():
    from dgl.data import DBLPDataset
    
    dataset = DBLPDataset()
    g = dataset[0]
    
    # 获取标签
    labels = g.nodes['paper'].data['label']
    
    # 创建划分
    num_papers = g.num_nodes('paper')
    indices = torch.randperm(num_papers)
    train_size = int(0.8 * num_papers)
    val_size = int(0.1 * num_papers)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:]
    
    return g, labels, train_idx, val_idx, test_idx

# 3. 创建合成数据集
def create_synthetic_dataset():
    # 创建一个小型合成异构图
    num_papers = 100
    num_authors = 50
    num_subjects = 10
    feature_dim = 64  # 使用较小的特征维度
    num_classes = 3
    
    # 创建边
    paper_author_src = np.random.randint(0, num_papers, 200)
    paper_author_dst = np.random.randint(0, num_authors, 200)
    
    paper_paper_src = np.random.randint(0, num_papers, 300)
    paper_paper_dst = np.random.randint(0, num_papers, 300)
    
    paper_subject_src = np.random.randint(0, num_papers, 150)
    paper_subject_dst = np.random.randint(0, num_subjects, 150)
    
    # 创建DGL图
    G = dgl.heterograph({
        ('paper', 'written-by', 'author'): (paper_author_src, paper_author_dst),
        ('author', 'writing', 'paper'): (paper_author_dst, paper_author_src),
        ('paper', 'citing', 'paper'): (paper_paper_src, paper_paper_dst),
        ('paper', 'cited', 'paper'): (paper_paper_dst, paper_paper_src),
        ('paper', 'is-about', 'subject'): (paper_subject_src, paper_subject_dst),
        ('subject', 'has', 'paper'): (paper_subject_dst, paper_subject_src),
    })
    
    # 创建标签
    labels = torch.randint(0, num_classes, (num_papers,)).long()
    
    # 创建划分
    indices = torch.randperm(num_papers)
    train_idx = indices[:80]
    val_idx = indices[80:90]
    test_idx = indices[90:]
    
    return G, labels, train_idx, val_idx, test_idx

# 根据选择加载数据集
if args.dataset == 'acm':
    G, labels, train_idx, val_idx, test_idx = load_acm_dataset(args.local_path)
    feature_dim = 400  # ACM使用原始特征维度
elif args.dataset == 'dblp':
    G, labels, train_idx, val_idx, test_idx = load_dblp_dataset()
    feature_dim = 64  # DBLP使用较小的特征维度
elif args.dataset == 'synthetic':
    G, labels, train_idx, val_idx, test_idx = create_synthetic_dataset()
    feature_dim = 64  # 合成数据集使用较小的特征维度
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")

print(G)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 设置节点和边类型字典
G.node_dict = {}
G.edge_dict = {}
for ntype in G.ntypes:
    G.node_dict[ntype] = len(G.node_dict)
for etype in G.etypes:
    G.edge_dict[etype] = len(G.edge_dict)
    G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * G.edge_dict[etype]
    
G = G.to(device)

# 随机初始化节点特征
for ntype in G.ntypes:
    emb = nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), feature_dim), requires_grad=False)
    nn.init.xavier_uniform_(emb)
    G.nodes[ntype].data['inp'] = emb

# for ntype in G.ntypes:
#     emb = nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), 400), requires_grad = False).to(device)
#     nn.init.xavier_uniform_(emb)
#     G.nodes[ntype].data['inp'] = emb

# 原生PyTorch实现的HGT层
class NativeHGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout=0.2, use_norm=False):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = num_types
        self.num_relations = num_relations
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        
        # 线性变换
        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm
        
        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        
        # 关系参数
        self.relation_pri = nn.Parameter(torch.ones(num_relations, n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(num_types))
        self.drop = nn.Dropout(dropout)
        
        # 初始化
        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)
    
    def forward(self, G, h_dict, inp_key, out_key):
        """
        G: DGL异构图
        h_dict: 保存节点特征的字典
        inp_key: 输入特征的键
        out_key: 输出特征的键
        """
        # 初始化节点特征（如果不在图中）
        for ntype in G.ntypes:
            if inp_key not in G.nodes[ntype].data:
                G.nodes[ntype].data[inp_key] = h_dict[ntype]
        
        # 特征变换
        for srctype, etype, dsttype in G.canonical_etypes:
            k_linear = self.k_linears[G.node_dict[srctype]]
            v_linear = self.v_linears[G.node_dict[srctype]]
            q_linear = self.q_linears[G.node_dict[dsttype]]
            
            # 计算k, q, v
            G.nodes[srctype].data['k'] = k_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[srctype].data['v'] = v_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[dsttype].data['q'] = q_linear(G.nodes[dsttype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            
            # 手动应用边特定的注意力（替代G.apply_edges）
            rel_id = G.edge_dict[etype]
            relation_att = self.relation_att[rel_id]
            relation_pri = self.relation_pri[rel_id]
            relation_msg = self.relation_msg[rel_id]
            
            # 获取边的端点
            src, dst = G.edges(etype=etype)
            
            # 获取源节点和目标节点的特征
            k_src = G.nodes[srctype].data['k'][src]  # [E, H, D]
            q_dst = G.nodes[dsttype].data['q'][dst]  # [E, H, D]
            v_src = G.nodes[srctype].data['v'][src]  # [E, H, D]
            
            # 计算注意力得分
            k_trans = torch.einsum('ehd,hdk->ehk', k_src, relation_att)
            att = (q_dst * k_trans).sum(dim=-1) * relation_pri / self.sqrt_dk
            
            # 转换值
            v_trans = torch.einsum('ehd,hdk->ehk', v_src, relation_msg)
            
            # 保存到边
            G.edges[etype].data['a'] = att
            G.edges[etype].data['v'] = v_trans
        
        # 手动执行消息聚合（替代G.multi_update_all）
        for ntype in G.ntypes:
            # 初始化结果张量
            G.nodes[ntype].data['t'] = torch.zeros(
                G.number_of_nodes(ntype), self.out_dim, device=device
            )
            
            # 计数器，用于计算平均值
            G.nodes[ntype].data['z'] = torch.zeros(
                G.number_of_nodes(ntype), 1, device=device
            )
        
        # 对每种边类型执行消息聚合
        for srctype, etype, dsttype in G.canonical_etypes:
            # 获取边的端点
            src, dst = G.edges(etype=etype)
            
            # 获取注意力权重和值
            a = G.edges[etype].data['a']  # [E, H]
            v = G.edges[etype].data['v']  # [E, H, D]
            
            # 对目标节点应用注意力
            for i in range(G.number_of_nodes(dsttype)):
                idx = (dst == i).nonzero(as_tuple=True)[0]
                if len(idx) == 0:
                    continue
                
                # 获取当前节点的边
                a_i = a[idx]  # [E_i, H]
                v_i = v[idx]  # [E_i, H, D]
                
                # 应用softmax
                a_i = F.softmax(a_i, dim=0)  # [E_i, H]
                
                # 加权聚合
                weighted_v = (a_i.unsqueeze(-1) * v_i).sum(dim=0)  # [H, D]
                
                # 将结果添加到节点
                G.nodes[dsttype].data['t'][i] = weighted_v.view(-1)
                G.nodes[dsttype].data['z'][i] += 1
        
        # 最终处理和残差连接
        for ntype in G.ntypes:
            n_id = G.node_dict[ntype]
            alpha = torch.sigmoid(self.skip[n_id])
            
            # 应用注意力结果
            trans_out = self.a_linears[n_id](G.nodes[ntype].data['t'])
            
            # 残差连接
            trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1-alpha)
            
            # 归一化和Dropout
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[n_id](trans_out))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)
        
        # 返回结果
        h_dict = {ntype: G.nodes[ntype].data[out_key] for ntype in G.ntypes}
        return h_dict

# 完整的HGT模型
class NativeHGT(nn.Module):
    def __init__(self, G, n_inp, n_hid, n_out, n_layers, n_heads, use_norm=True):
        super().__init__()
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        
        # 输入投影
        self.adapt_ws = nn.ModuleList()
        for t in range(len(G.node_dict)):
            self.adapt_ws.append(nn.Linear(n_inp, n_hid))
        
        # HGT层
        for _ in range(n_layers):
            self.gcs.append(NativeHGTLayer(
                n_hid, n_hid, len(G.node_dict), len(G.edge_dict), n_heads, use_norm=use_norm
            ))
        
        # 输出层
        self.out = nn.Linear(n_hid, n_out)
    
    def forward(self, G, out_key):
        # 初始化特征字典
        h_dict = {}
        for ntype in G.ntypes:
            n_id = G.node_dict[ntype]
            h_dict[ntype] = torch.tanh(self.adapt_ws[n_id](G.nodes[ntype].data['inp']))
            G.nodes[ntype].data['h'] = h_dict[ntype]
        
        # 应用HGT层
        for i in range(self.n_layers):
            h_dict = self.gcs[i](G, h_dict, 'h', 'h')
        
        # 返回目标节点类型的预测
        return self.out(h_dict[out_key])

# 实例化模型
hidden_dim = 128 if args.dataset != 'acm' else 200  # 对于较小的数据集使用较小的隐藏维度
model = NativeHGT(
    G, n_inp=feature_dim, n_hid=hidden_dim, n_out=labels.max().item()+1, 
    n_layers=2, n_heads=4, use_norm=True
).to(device)

# 将标签移动到设备
labels = labels.to(device)
train_idx = train_idx.to(device)
val_idx = val_idx.to(device)
test_idx = test_idx.to(device)

optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, total_steps=100, max_lr=1e-3, pct_start=0.05
)

# 训练循环
best_val_acc = 0
best_test_acc = 0
train_step = 0
for epoch in range(100):
    logits = model(G, 'paper')
    # 损失计算
    loss = F.cross_entropy(logits[train_idx], labels[train_idx])

    pred = logits.argmax(1).cpu()
    train_acc = (pred[train_idx.cpu()] == labels.cpu()[train_idx.cpu()]).float().mean()
    val_acc = (pred[val_idx.cpu()] == labels.cpu()[val_idx.cpu()]).float().mean()
    test_acc = (pred[test_idx.cpu()] == labels.cpu()[test_idx.cpu()]).float().mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_step += 1
    scheduler.step(train_step)

    if best_val_acc < val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
    
    if epoch % 1 == 0:
        print('[%s] Epoch %d | LR: %.5f Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
            args.dataset,
            epoch,
            optimizer.param_groups[0]['lr'], 
            loss.item(),
            train_acc.item(),
            val_acc.item(),
            best_val_acc.item(),
            test_acc.item(),
            best_test_acc.item(),
        ))
