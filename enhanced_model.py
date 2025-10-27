import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DebertaV2Model
import math

class MultiHeadDebiasedAttention(nn.Module):
    """
    去偏注意力机制
    """
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # 去偏参数
        self.bias_correction = nn.Parameter(torch.zeros(num_heads))
        
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, hidden_size = hidden_states.size()
        
        # 计算Q, K, V
        Q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 去偏校正：减去每个头的偏置
        bias_correction = self.bias_correction.view(1, -1, 1, 1)
        scores = scores - bias_correction
        
        # 应用注意力掩码
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # 计算注意力权重
        attention_probs = F.softmax(scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 加权求和
        context = torch.matmul(attention_probs, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        # 输出投影
        output = self.out_proj(context)
        return output, attention_probs


class MultiLevelFeatureExtractor(nn.Module):
    """
    多层次特征提取器
    """
    def __init__(self, hidden_size, num_layers=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 多层次特征提取
        self.layer_weights = nn.Parameter(torch.ones(num_layers))
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_size * num_layers, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, all_hidden_states):
        """
        Args:
            all_hidden_states: List of hidden states from different layers
        """
        # 获取最后几层的隐藏状态
        selected_layers = all_hidden_states[-self.num_layers:]
        
        # 加权融合不同层的特征
        weighted_layers = []
        layer_weights = F.softmax(self.layer_weights, dim=0)
        
        for i, hidden_state in enumerate(selected_layers):
            cls_hidden = hidden_state[:, 0, :]  # [CLS] token
            weighted_layers.append(cls_hidden * layer_weights[i])
        
        # 拼接多层特征
        multi_level_features = torch.cat(weighted_layers, dim=-1)
        
        # 特征融合
        fused_features = self.feature_fusion(multi_level_features)
        return fused_features


class EnhancedTimeProjector(nn.Module):
    """
    增强版时间特征投影器
    """
    def __init__(self, time_feature_dim, hidden_dim, num_layers=2):
        super().__init__()
        
        layers = []
        in_dim = time_feature_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim if i == num_layers - 1 else hidden_dim // 2
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = out_dim
        
        # 移除最后的激活和dropout
        self.projector = nn.Sequential(*layers[:-2])
        
        # 注意力机制用于时间特征加权
        self.time_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, time_features):
        projected = self.projector(time_features)
        
        # 计算时间特征的重要性权重
        attention_weights = self.time_attention(projected)
        weighted_features = projected * attention_weights
        
        return weighted_features


class EnhancedSuicideRiskClassifier(nn.Module):
    """
    增强版自杀风险分类器
    - 去偏注意力机制
    - 多层次特征提取
    - 增强时间特征融合
    - 相对位置编码增强
    """
    
    def __init__(self, model_name='microsoft/deberta-v3-large', 
                 hidden_dim=512, num_labels=4, time_feature_dim=12,
                 pattern_feature_dim=6, use_debiased_attention=True):
        super().__init__()
        
        # 文本编码器
        self.bert = DebertaV2Model.from_pretrained(model_name, output_hidden_states=True)
        bert_hidden_size = self.bert.config.hidden_size
        
        # 冻结前几层（可选）
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i in range(6):  # 冻结前6层
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
        
        # GRU处理帖子序列
        self.rnn = nn.GRU(
            bert_hidden_size, 
            hidden_dim, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.1 if hidden_dim > 1 else 0
        )
        
        # 去偏注意力机制（可选）
        self.use_debiased_attention = use_debiased_attention
        if use_debiased_attention:
            self.debiased_attention = MultiHeadDebiasedAttention(
                hidden_size=bert_hidden_size,
                num_heads=8,
                dropout=0.1
            )
        
        # 多层次特征提取器
        self.multi_level_extractor = MultiLevelFeatureExtractor(
            hidden_size=bert_hidden_size,
            num_layers=3
        )
        
        # 增强时间特征投影
        self.time_proj = EnhancedTimeProjector(
            time_feature_dim=time_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=2
        )
        
        # 时间模式特征投影
        self.pattern_proj = nn.Sequential(
            nn.Linear(pattern_feature_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # 特征融合层
        fusion_input_dim = hidden_dim * 2 + hidden_dim + hidden_dim // 2  # text + time + pattern
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_labels)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化新增层的权重"""
        for module in [self.feature_fusion, self.classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)

    def forward(self, input_ids, attention_mask, time_features, pattern_features=None):
        B, T, L = input_ids.size()  # (batch, max_posts, max_len)
        
        # 重塑输入以处理多个帖子
        input_ids = input_ids.view(B * T, L)
        attention_mask = attention_mask.view(B * T, L)

        # 文本编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 应用去偏注意力（可选）
        if self.use_debiased_attention:
            debiased_output, _ = self.debiased_attention(
                outputs.last_hidden_state, 
                attention_mask
            )
            # 残差连接
            enhanced_hidden = outputs.last_hidden_state + debiased_output
        else:
            enhanced_hidden = outputs.last_hidden_state
        
        # 多层次特征提取
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            multi_level_cls = self.multi_level_extractor(outputs.hidden_states)
            multi_level_cls = multi_level_cls.view(B, T, -1)
        else:
            # 备用方案：使用CLS token
            cls_embeddings = enhanced_hidden[:, 0, :]  # (B*T, H)
            multi_level_cls = cls_embeddings.view(B, T, -1)  # (B, T, H)

        # GRU处理帖子序列
        _, hidden = self.rnn(multi_level_cls)  # hidden: (2, B, hidden_dim)
        text_features = torch.cat([hidden[0], hidden[1]], dim=-1)  # (B, 2*hidden_dim)
        
        # 时间特征处理
        time_emb = self.time_proj(time_features)  # (B, hidden_dim)
        
        # 时间模式特征处理
        if pattern_features is not None:
            pattern_emb = self.pattern_proj(pattern_features)  # (B, hidden_dim//2)
            combined_features = torch.cat([text_features, time_emb, pattern_emb], dim=-1)
        else:
            # 如果没有模式特征，用零填充
            pattern_emb = torch.zeros(B, time_emb.size(-1) // 2, device=time_emb.device)
            combined_features = torch.cat([text_features, time_emb, pattern_emb], dim=-1)
        
        # 特征融合
        fused_features = self.feature_fusion(combined_features)
        
        # 分类
        logits = self.classifier(fused_features)
        return logits


# 保持向后兼容的原版模型
class SuicideRiskClassifier(nn.Module):
    """原版模型（向后兼容）"""
    def __init__(self, model_name='microsoft/deberta-v3-large', 
                 hidden_dim=512, num_labels=4, time_feature_dim=8):
        super().__init__()
        self.text_encoder = DebertaV2Model.from_pretrained(model_name)
        text_hidden_size = self.text_encoder.config.hidden_size

        # 时间特征映射到同一空间
        self.time_proj = nn.Sequential(
            nn.Linear(time_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 融合层
        self.classifier = nn.Sequential(
            nn.Linear(text_hidden_size + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, input_ids, attention_mask, time_features):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = text_outputs.last_hidden_state[:, 0, :]  # [CLS]

        time_emb = self.time_proj(time_features)

        combined = torch.cat([cls_embedding, time_emb], dim=1)
        logits = self.classifier(combined)

        return logits