import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GCNConv, GATConv


class PositionalEncoding(nn.Module):
    """位置编码模块，为时序数据添加位置信息"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        # 初始化位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 动态扩展位置编码以适应更长的序列
        if x.size(1) > self.max_len:
            # 计算需要扩展的长度
            new_len = x.size(1)

            # 创建新的位置编码矩阵
            new_pe = torch.zeros(new_len, self.pe.size(2), device=x.device)
            position = torch.arange(0, new_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.pe.size(2), 2).float() *
                                 (-math.log(10000.0) / self.pe.size(2))).to(x.device)
            new_pe[:, 0::2] = torch.sin(position * div_term)
            new_pe[:, 1::2] = torch.cos(position * div_term)
            new_pe = new_pe.unsqueeze(0)

            # 使用新的位置编码
            x = x + new_pe[:, :x.size(1)]
        else:
            # 使用预设的位置编码
            x = x + self.pe[:, :x.size(1)]

        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """增强的Transformer编码层，添加卷积门控机制"""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # 添加卷积门控机制
        self.conv_gate = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 使用GELU激活函数
        self.activation = nn.GELU()

    def forward(self, src):
        # 自注意力机制
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 卷积门控机制
        gate = self.conv_gate(src.permute(0, 2, 1)).permute(0, 2, 1)
        src_gated = src * gate

        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_gated))))
        src = src + self.dropout2(src2)
        return self.norm2(src)


class UncertaintyAwareScoring(nn.Module):
    """Heteroscedastic binary scoring head.

    ``mean_layer`` predicts the latent anomaly logit and ``var_layer`` predicts
    its aleatoric variance.  The variance is consumed by the training objective
    and by the logistic-normal predictive probability used at evaluation time.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.mean_layer = nn.Linear(hidden_dim, 1)
        self.var_layer = nn.Linear(hidden_dim, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        # Both outputs have shape [num_timesteps, 1].
        mean = self.mean_layer(x)
        # A positive, bounded variance avoids invalid square roots and extreme
        # Monte-Carlo logits while retaining a learnable uncertainty signal.
        var = (self.softplus(self.var_layer(x)) + 1e-6).clamp(max=25.0)
        return mean, var


class TransformerEncoder(nn.Module):
    """Transformer编码器，使用增强的编码层"""

    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src):
        output = src
        for layer in self.layers:
            output = layer(output)
        return self.norm(output)


class GraphBranch(nn.Module):
    """增强的图神经网络分支，确保处理各种边界情况（适配时间点级图结构）"""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 图卷积层
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # 图注意力层
        self.gat = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)

    def forward(self, data):
        """Apply three GCN layers and two-head GAT without masking failures."""
        x, edge_index = data.x, data.edge_index
        if x.ndim != 2 or x.size(0) == 0:
            raise ValueError("Graph must contain at least one node")
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= x.size(0)):
            raise ValueError("edge_index contains an out-of-range node")
        x = F.dropout(F.gelu(self.conv1(x, edge_index)), p=0.3, training=self.training)
        x = F.dropout(F.gelu(self.conv2(x, edge_index)), p=0.3, training=self.training)
        x = F.gelu(self.conv3(x, edge_index))
        return self.gat(x, edge_index)


class FeatureSelector(nn.Module):
    """特征选择层，自动学习特征重要性"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.selector = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.selector(x)
        return x * weights

class TemporalBranch(nn.Module):
    """增强的时序分支，处理每个时间点内的多个数据包（已聚合）"""

    def __init__(self, input_dim, config):
        super().__init__()
        self.config = config

        # 数据包级特征提取（现在每个时间点只有一个特征向量）
        self.packet_embed = nn.Linear(input_dim, config['hidden_dim'])

        # 位置编码
        self.pos_encoder = PositionalEncoding(config['hidden_dim'])

        # 时间点级Transformer
        self.time_transformer = TransformerEncoder(
            d_model=config['hidden_dim'],
            nhead=config['num_heads'],
            dim_feedforward=config['d_ff'],
            num_layers=config['e_layers'],
            dropout=config['dropout']
        )

    def forward(self, data):
        # 获取时间点级特征
        x = data.x  # 形状为 [num_timesteps, feature_dim]

        # 数据包嵌入
        x_emb = self.packet_embed(x)

        # 应用位置编码
        x_emb = self.pos_encoder(x_emb.unsqueeze(0)).squeeze(0)

        # 时间点级Transformer
        time_out = self.time_transformer(x_emb.unsqueeze(0)).squeeze(0)

        return time_out

class GatedFusion(nn.Module):
    """门控融合模块，用于更有效地融合时序和空间特征"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.temporal_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, temporal, spatial):
        # 确保两个输入张量在维度0上大小一致
        if temporal.size(0) != spatial.size(0):
            # 如果大小不匹配，调整空间特征的大小
            if spatial.size(0) == 1:
                # 如果空间特征只有1个，复制到时序特征的数量
                spatial = spatial.expand(temporal.size(0), -1)
            else:
                # 否则截断或填充
                min_size = min(temporal.size(0), spatial.size(0))
                temporal = temporal[:min_size]
                spatial = spatial[:min_size]

        temporal_gate = self.temporal_gate(temporal)
        spatial_gate = self.spatial_gate(spatial)

        gated_temporal = temporal * temporal_gate
        gated_spatial = spatial * spatial_gate

        combined = torch.cat([gated_temporal, gated_spatial], dim=1)
        fused = self.fusion_layer(combined)
        return fused

class LightAnomalyDetector(nn.Module):
    """增强版异常检测模型，适配时间点级输入"""

    def __init__(self, input_dim, config):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.difficulty = 1.0

        # 特征选择层
        self.feature_selector = FeatureSelector(input_dim, input_dim)

        # 时序分支
        self.temporal_branch = TemporalBranch(input_dim, config)

        # 图分支 - 使用修改后的健壮版本
        self.graph_branch = GraphBranch(input_dim, config['hidden_dim'])

        # 特征融合
        self.fusion = GatedFusion(config['hidden_dim'])

        # 不确定性评分层
        self.scoring = UncertaintyAwareScoring(config['hidden_dim'])

        # 难度适应层
        self.difficulty_adapter = nn.Linear(config['hidden_dim'], config['hidden_dim'])

    def forward(self, data):
        # 特征选择
        data = data.clone()
        data.x = self.feature_selector(data.x)

        # 时序特征
        temporal_out = self.temporal_branch(data)

        # 图特征
        graph_out = self.graph_branch(data)

        # 确保两个特征在维度上匹配
        if temporal_out.size(0) != graph_out.size(0):
            min_size = min(temporal_out.size(0), graph_out.size(0))
            temporal_out = temporal_out[:min_size]
            graph_out = graph_out[:min_size]

        # 特征融合
        fused = self.fusion(temporal_out, graph_out)

        # 难度适应
        fused = self.difficulty_adapter(fused)

        # 异常评分
        mean, var = self.scoring(fused)
        return mean, var


# Public name; historical class name remains available for source compatibility.
STDNet = LightAnomalyDetector
