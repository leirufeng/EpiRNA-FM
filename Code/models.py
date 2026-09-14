import math
import torch
import torch.nn as nn
import torch.nn.functional as F



# ===================== EfficientNet1D =====================

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            Swish(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):

        b, c, _ = x.size()

        y = self.pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1)

        return x * y


class MBConv1D(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            expand_ratio=4,
            kernel_size=3,
            stride=1):

        super().__init__()

        hidden_dim = in_channels * expand_ratio

        self.use_residual = (
            stride == 1 and in_channels == out_channels
        )

        # Expansion
        self.expand = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            Swish()
        )

        # Depthwise Conv
        self.depthwise = nn.Sequential(
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=hidden_dim,
                bias=False
            ),
            nn.BatchNorm1d(hidden_dim),
            Swish()
        )

        # SE Attention
        self.se = SEBlock(hidden_dim)

        # Projection
        self.project = nn.Sequential(
            nn.Conv1d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):

        identity = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.project(x)
        if self.use_residual:
            x = x + identity

        return x


class EfficientNet1D(nn.Module):

    def __init__(self, in_channels=4):

        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            Swish()
        )

        # MBConv Stages
        self.blocks = nn.Sequential(
            MBConv1D(64, 64, expand_ratio=1),
            MBConv1D(64, 96, expand_ratio=4),
            MBConv1D(96, 192, expand_ratio=4),
            # MBConv1D(128, 192, expand_ratio=6),
            # MBConv1D(192, 192, expand_ratio=6)
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        x = self.stem(x)

        x = self.blocks(x)

        x = self.pool(x).squeeze(-1)

        return x


# =====================================================
# Transformer Encoder
# =====================================================
class TransformerEncoder(nn.Module):
    def __init__(self, dim=64, out_dim=192, max_len=5000, num_layers=2):
        
        super().__init__()
        self.embedding = nn.Embedding(5, dim)   

        # Position Embedding
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)          # shape: (1, max_len, dim)
        self.register_buffer('pe', pe)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            batch_first=True,
            dropout=0.1
        )
        self.trans = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        
        self.fc = nn.Linear(dim, out_dim)

    def forward(self, x):
        
        seq_len = x.size(1)
        
        if seq_len > self.pe.size(1):
            x = x[:, :self.pe.size(1)]
            seq_len = self.pe.size(1)

       
        x = self.embedding(x) + self.pe[:, :seq_len, :]   # (batch, seq_len, dim)
        
        x = self.trans(x)   # (batch, seq_len, dim)

        
        x = x.mean(dim=1)   # (batch, dim)

        
        return self.fc(x)

# =====================================================
# Cross Attention
# =====================================================

class BiCrossAttention(nn.Module):
    def __init__(self, dim=192):
        super().__init__()

        self.attn1 = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.attn2 = nn.MultiheadAttention(dim, 8, batch_first=True)

    def forward(self, s, t):

        s = s.unsqueeze(1)
        t = t.unsqueeze(1)

        s2, _ = self.attn1(s, t, t)

        t2, _ = self.attn2(t, s, s)

        return (s2.squeeze(1), t2.squeeze(1))


# =====================================================
# CM-SE
# =====================================================

class CMSE(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.fc = nn.Sequential(

            nn.Linear(
                dim * 2,
                dim
            ),

            nn.GELU(),

            nn.Linear(
                dim,
                dim * 2
            ),

            nn.Sigmoid()
        )

    def forward(self, s, t):

        w = self.fc(
            torch.cat([s, t], dim=1)
        )

        ws, wt = torch.chunk(
            w,
            2,
            dim=1
        )

        return s * ws, t * wt


# =====================================================
# Shared Encoder
# =====================================================

class SharedEncoder(nn.Module):
    def __init__(self):

        super().__init__()

        self.cnn = EfficientNet1D()
        self.trans = TransformerEncoder()
        self.cross = BiCrossAttention(192)
        self.cmse = CMSE(192)

    def forward(self, x1, x2):

        cnn_feat = self.cnn(x1)
        trans_feat = self.trans(x2)

        cnn_feat, trans_feat = self.cross(cnn_feat, trans_feat)

        cnn_feat, trans_feat = self.cmse(cnn_feat, trans_feat)

        feat = torch.cat([cnn_feat, trans_feat], dim=1)

        return feat

        # [B,384]


# =====================================================
# Prototype Interface
# =====================================================

class PrototypeLayer(nn.Module):

    def __init__(self, num_tasks=12, feat_dim=384):
        super().__init__()

        self.prototype = nn.Parameter(torch.randn(num_tasks, feat_dim))

    def forward(self):

        return self.prototype


# =====================================================
# Multi Task Head
# =====================================================

class MultiTaskHead(nn.Module):
    def __init__(self, feat_dim=384, num_tasks=12):
        super().__init__()

        self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feat_dim, 256),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 2))
                for _ in range(num_tasks)
            ])

    def forward(self, feat, task_ids):

        logits = []

        for i in range(len(task_ids)):

            task = task_ids[i]
            logit = self.heads[task](feat[i].unsqueeze(0))
            logits.append(logit)

        logits = torch.cat(logits, dim=0)

        return logits


# =====================================================
# Full Model
# =====================================================

class RNAFoundationModel(nn.Module):
    def __init__(self, num_tasks=12):
        super().__init__()

        self.encoder = SharedEncoder()

        self.prototype_layer = PrototypeLayer(num_tasks=num_tasks, feat_dim=384)

        self.task_heads = MultiTaskHead(feat_dim=384, num_tasks=num_tasks)

    def forward(self, x1, x2, task_ids):

        feat = self.encoder(x1, x2)

        logits = self.task_heads(feat, task_ids)

        return {"feature": feat, "prototype": self.prototype_layer(), "logits": logits}


# =====================================================
# Debug
# =====================================================

if __name__ == "__main__":

    model = RNAFoundationModel()

    x1 = torch.randn(1, 4, 101)

    x2 = torch.randint(0, 5, (1,101))

    task_ids = torch.randint(0, 12, (1,))

    out = model(x1, x2, task_ids)

    print(out["feature"].shape)

    print(out["prototype"].shape)

    print(out["logits"].shape)