import torch
import torch.nn as nn
import torch.nn.functional as F


class BiologicalPriorGraph(nn.Module):
    def __init__(self, num_tasks=12):
        super().__init__()
        adj = torch.eye(num_tasks)
        
        families = [
            [0, 1, 4, 7, 8],   
            [2, 5],            
            [3, 9],            
            [6, 10, 11]        
        ]
        for fam in families:
            for i in fam:
                for j in fam:
                    if i != j:
                        adj[i, j] = 1
        self.register_buffer("adj", adj)  # [num_tasks, num_tasks], 0/1

    def get_adj(self):
        return self.adj


class PrototypeMemory(nn.Module):
    def __init__(self, num_tasks=12, feat_dim=384, momentum=0.9):
        super().__init__()
        self.num_tasks = num_tasks
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.register_buffer("prototype_pos", torch.zeros(num_tasks, feat_dim))
        self.register_buffer("prototype_neg", torch.zeros(num_tasks, feat_dim))

    @torch.no_grad()
    def update(self, features, task_ids, labels):
        unique_tasks = torch.unique(task_ids)
        for task in unique_tasks:
            mask = (task_ids == task)
            if mask.sum() == 0:
                continue
            task_feat = features[mask]
            task_label = labels[mask]

            pos_mask = (task_label == 1)
            if pos_mask.sum() > 0:
                mean_pos = task_feat[pos_mask].mean(0)
                self.prototype_pos[task] = self.momentum * self.prototype_pos[task] + (1 - self.momentum) * mean_pos

            neg_mask = (task_label == 0)
            if neg_mask.sum() > 0:
                mean_neg = task_feat[neg_mask].mean(0)
                self.prototype_neg[task] = self.momentum * self.prototype_neg[task] + (1 - self.momentum) * mean_neg

    def get_prototypes(self):
        return self.prototype_pos, self.prototype_neg


class TaskRelationGraph(nn.Module):
    def __init__(self, prior_adj, mode='hard_mask', alpha=0.5, temperature=1.0):
        
        super().__init__()
        self.register_buffer("prior_adj", prior_adj)
        self.mode = mode
        self.alpha = alpha
        self.temperature = temperature

       
        if mode == 'interpolate':
            
            prior_norm = F.softmax(prior_adj.float(), dim=1)
            self.register_buffer("prior_norm", prior_norm)

    def forward(self, prototype_pos):
        
        
        proto_norm = F.normalize(prototype_pos, dim=1)
        sim = torch.mm(proto_norm, proto_norm.t())  

        if self.mode == 'hard_mask':
            
            masked_sim = torch.where(
                self.prior_adj > 0,
                sim / self.temperature,
                torch.full_like(sim, -float('inf'))
            )
            adjacency = F.softmax(masked_sim, dim=1)
            
            row_nan = torch.isnan(adjacency).any(dim=1)
            if row_nan.any():
                adjacency[row_nan] = torch.eye(adjacency.size(0), device=adjacency.device)[row_nan]
            return adjacency

        elif self.mode == 'interpolate':
            
            dyn_adj = F.softmax(sim / self.temperature, dim=1)
            adjacency = self.alpha * self.prior_norm + (1 - self.alpha) * dyn_adj
            
            return adjacency

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(out_dim * 2, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, h, adj):
        Wh = self.W(h)
        N = Wh.size(0)
        a_input = torch.cat([Wh.repeat(1, N).view(N*N, -1), Wh.repeat(N, 1)], dim=1).view(N, N, -1)
        e = self.leakyrelu(self.a(a_input).squeeze(-1))
        
        attention = torch.where(adj > 0, e, torch.full_like(e, -9e15))
        attention = F.softmax(attention, dim=1)
        h_prime = torch.matmul(attention, Wh)
        return h_prime

class MultiHeadGAT(nn.Module):
    def __init__(self, feat_dim=384, num_heads=4):
        super().__init__()
        self.heads = nn.ModuleList([
            GraphAttentionLayer(feat_dim, feat_dim) for _ in range(num_heads)
        ])
        self.fc = nn.Linear(feat_dim * num_heads, feat_dim)

    def forward(self, prototype, adjacency):
        outs = [head(prototype, adjacency) for head in self.heads]
        out = torch.cat(outs, dim=1)
        out = self.fc(out)
        return out


class TaskGraphModule(nn.Module):
    def __init__(self, num_tasks=12, feat_dim=384,
                 prior_mode='hard_mask', prior_alpha=0.5, temperature=1.0):
        super().__init__()
        
        self.prior_graph = BiologicalPriorGraph(num_tasks)
        prior_adj = self.prior_graph.get_adj()

        
        self.memory = PrototypeMemory(num_tasks, feat_dim)

       
        self.graph = TaskRelationGraph(
            prior_adj,
            mode=prior_mode,
            alpha=prior_alpha,
            temperature=temperature
        )

        
        self.gat = MultiHeadGAT(feat_dim=feat_dim)

    def forward(self, features=None, task_ids=None, labels=None):
        
        if features is not None and task_ids is not None and labels is not None:
            self.memory.update(features, task_ids, labels)

        prototype_pos, prototype_neg = self.memory.get_prototypes()

        
        adjacency = self.graph(prototype_pos)

        
        graph_proto = self.gat(prototype_pos, adjacency)

        return {
            "prototype_pos": prototype_pos,
            "prototype_neg": prototype_neg,
            "adjacency": adjacency,        
            "graph_proto": graph_proto
        }