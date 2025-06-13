import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import math

class ComplexityAwareAttention(nn.Module):
    """复杂度感知注意力机制"""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 复杂度惩罚参数
        self.complexity_penalty = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                complexity: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        复杂度感知注意力前向传播
        
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            complexity: [batch_size, seq_len] - 复杂度分数
            mask: 可选的掩码
        """
        batch_size, seq_len, _ = query.size()
        
        # 线性变换
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用复杂度惩罚
        complexity_penalty = self.complexity_penalty * complexity.unsqueeze(1).unsqueeze(1)
        scores = scores - complexity_penalty.expand_as(scores)
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        
        # 输出投影
        output = self.w_o(context)
        
        # 残差连接和层归一化
        output = self.layer_norm(output + query)
        
        return output

class ComplexityEncoder(nn.Module):
    """复杂度编码器"""
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # 输出0-1之间的复杂度分数
        )
    
    def forward(self, kolmogorov: torch.Tensor, computational: torch.Tensor, 
                logical_depth: torch.Tensor) -> torch.Tensor:
        """
        编码复杂度特征
        
        Args:
            kolmogorov: Kolmogorov复杂度
            computational: 计算复杂度（对数）
            logical_depth: 逻辑深度
        """
        # 归一化输入
        computational_norm = torch.log(computational + 1) / 10.0
        logical_depth_norm = logical_depth / 20.0
        
        # 组合特征
        complexity_features = torch.stack([
            kolmogorov, computational_norm, logical_depth_norm
        ], dim=-1)
        
        # 添加交叉项
        cross_terms = torch.stack([
            kolmogorov * computational_norm,
            kolmogorov * logical_depth_norm,
            computational_norm * logical_depth_norm,
            kolmogorov * computational_norm * logical_depth_norm
        ], dim=-1)
        
        # 合并所有特征
        all_features = torch.cat([complexity_features, cross_terms], dim=-1)
        
        return self.mlp(all_features)

class InformationEncoder(nn.Module):
    """信息编码器"""
    
    def __init__(self, vocab_size: int = 10000, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        
        # 文本编码器
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = self._create_position_encoding(1000, d_model)
        
        # 数值编码器
        self.numerical_encoder = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model)
        )
        
        # 结构化数据编码器
        self.structured_encoder = nn.Sequential(
            nn.Linear(10, d_model // 2),  # 假设最多10个字段
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
    
    def _create_position_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """创建位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, text_ids: Optional[torch.Tensor] = None,
                numerical_data: Optional[torch.Tensor] = None,
                structured_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        编码不同类型的信息
        """
        encodings = []
        
        if text_ids is not None:
            text_emb = self.text_embedding(text_ids)
            seq_len = text_emb.size(1)
            pos_enc = self.position_encoding[:, :seq_len, :].to(text_emb.device)
            text_encoding = text_emb + pos_enc
            encodings.append(text_encoding)
        
        if numerical_data is not None:
            num_encoding = self.numerical_encoder(numerical_data.unsqueeze(-1))
            encodings.append(num_encoding.unsqueeze(1))
        
        if structured_data is not None:
            struct_encoding = self.structured_encoder(structured_data)
            encodings.append(struct_encoding.unsqueeze(1))
        
        if not encodings:
            raise ValueError("至少需要提供一种类型的输入数据")
        
        # 合并所有编码
        if len(encodings) == 1:
            return encodings[0]
        else:
            # 简单拼接，实际应用中可能需要更复杂的融合策略
            return torch.cat(encodings, dim=1)

class TemporalFusion(nn.Module):
    """时间融合模块"""
    
    def __init__(self, d_model: int = 256, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(d_model, d_model, num_layers, 
                           batch_first=True, dropout=0.1)
        self.attention = nn.MultiheadAttention(d_model, num_heads=8, dropout=0.1)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        时间序列融合
        
        Args:
            x: [batch_size, seq_len, d_model]
        """
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        
        # 自注意力
        x_transposed = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        attn_out, _ = self.attention(x_transposed, x_transposed, x_transposed)
        attn_out = attn_out.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # 残差连接
        output = self.layer_norm(lstm_out + attn_out + x)
        
        return output

class PriceDiscoveryPredictor(nn.Module):
    """价格发现预测器"""
    
    def __init__(self, d_model: int = 256, num_targets: int = 3):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, num_targets)
        )
        
        # 输出层：发现速度、效率、质量
        self.speed_head = nn.Linear(num_targets, 1)
        self.efficiency_head = nn.Linear(num_targets, 1)
        self.quality_head = nn.Linear(num_targets, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        预测价格发现指标
        
        Returns:
            speed: 发现速度
            efficiency: 发现效率
            quality: 调整质量
        """
        # 全局平均池化
        if x.dim() == 3:
            x = x.mean(dim=1)
        
        features = self.predictor(x)
        
        speed = torch.relu(self.speed_head(features))  # 速度必须为正
        efficiency = torch.sigmoid(self.efficiency_head(features))  # 效率在0-1之间
        quality = torch.sigmoid(self.quality_head(features))  # 质量在0-1之间
        
        return speed, efficiency, quality

class ComplexityAwareModel(nn.Module):
    """完整的复杂度感知模型"""
    
    def __init__(self, vocab_size: int = 10000, d_model: int = 256, 
                 n_heads: int = 8, num_layers: int = 6):
        super().__init__()
        
        # 组件初始化
        self.information_encoder = InformationEncoder(vocab_size, d_model)
        self.complexity_encoder = ComplexityEncoder(output_dim=d_model//8)
        self.temporal_fusion = TemporalFusion(d_model)
        self.price_discovery_predictor = PriceDiscoveryPredictor(d_model)
        
        # 复杂度感知注意力层
        self.attention_layers = nn.ModuleList([
            ComplexityAwareAttention(d_model, n_heads)
            for _ in range(num_layers)
        ])
        
    def forward(self, text_ids: Optional[torch.Tensor] = None,
                numerical_data: Optional[torch.Tensor] = None,
                structured_data: Optional[torch.Tensor] = None,
                kolmogorov: Optional[torch.Tensor] = None,
                computational: Optional[torch.Tensor] = None,
                logical_depth: Optional[torch.Tensor] = None) -> dict:
        """
        模型前向传播
        """
        # 信息编码
        info_encoding = self.information_encoder(text_ids, numerical_data, structured_data)
        
        # 复杂度编码
        if kolmogorov is not None and computational is not None and logical_depth is not None:
            complexity_scores = self.complexity_encoder(kolmogorov, computational, logical_depth)
            # 扩展复杂度分数到序列长度
            if complexity_scores.dim() == 2:
                complexity_scores = complexity_scores.unsqueeze(1).expand(-1, info_encoding.size(1), -1)
            complexity_penalty = complexity_scores.mean(dim=-1)  # [batch_size, seq_len]
        else:
            complexity_penalty = torch.zeros(info_encoding.size(0), info_encoding.size(1)).to(info_encoding.device)
        
        # 应用复杂度感知注意力层
        x = info_encoding
        for attention_layer in self.attention_layers:
            x = attention_layer(x, x, x, complexity_penalty)
        
        # 时间融合
        x = self.temporal_fusion(x)
        
        # 价格发现预测
        speed, efficiency, quality = self.price_discovery_predictor(x)
        
        return {
            'discovery_speed': speed,
            'discovery_efficiency': efficiency,
            'adjustment_quality': quality,
            'complexity_scores': complexity_penalty,
            'attention_features': x
        }

# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = ComplexityAwareModel(vocab_size=1000, d_model=128, n_heads=4, num_layers=3)
    
    # 创建测试数据
    batch_size = 8
    seq_len = 20
    
    text_ids = torch.randint(0, 1000, (batch_size, seq_len))
    numerical_data = torch.randn(batch_size, 5)
    structured_data = torch.randn(batch_size, 10)
    kolmogorov = torch.rand(batch_size)
    computational = torch.randint(1, 100, (batch_size,)).float()
    logical_depth = torch.randint(1, 10, (batch_size,)).float()
    
    # 前向传播
    with torch.no_grad():
        outputs = model(
            text_ids=text_ids,
            numerical_data=numerical_data,
            structured_data=structured_data,
            kolmogorov=kolmogorov,
            computational=computational,
            logical_depth=logical_depth
        )
    
    print("模型输出:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    print(f"\n发现速度样本: {outputs['discovery_speed'][:3].squeeze()}")
    print(f"发现效率样本: {outputs['discovery_efficiency'][:3].squeeze()}")
    print(f"调整质量样本: {outputs['adjustment_quality'][:3].squeeze()}")
