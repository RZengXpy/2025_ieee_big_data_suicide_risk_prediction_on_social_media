import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DebertaV2Model
import math
from enhanced_model import MultiHeadDebiasedAttention, MultiLevelFeatureExtractor, EnhancedTimeProjector

class CoralLoss(nn.Module):
    """
    CORAL损失函数 (Cumulative Ordinal Regression)
    """
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight
        
    def forward(self, logits, targets):
        """
        计算CORAL损失
        
        Args:
            logits: 模型输出，形状为 (batch_size, 3) - 3个二分类logits
            targets: 真实标签，形状为 (batch_size,) - 类别标签 0,1,2,3
        """
        batch_size = logits.size(0)
        
        # 将类别标签转换为累积标签
        # 对于类别0,1,2,3，转换为3个二分类标签
        # y=0 -> [0,0,0]
        # y=1 -> [1,0,0]
        # y=2 -> [1,1,0]
        # y=3 -> [1,1,1]
        targets_cum = torch.zeros(batch_size, 3, device=logits.device)
        targets_cum[:, 0] = (targets >= 1).float()  # P(y > 0)
        targets_cum[:, 1] = (targets >= 2).float()  # P(y > 1)
        targets_cum[:, 2] = (targets >= 3).float()  # P(y > 2)
        
        # 计算sigmoid概率
        probs = torch.sigmoid(logits)
        
        # 计算二元交叉熵损失
        if self.weight is not None:
            # 为每个样本应用权重
            sample_weights = self.weight[targets]  # 根据真实标签获取权重
            bce_loss = F.binary_cross_entropy(probs, targets_cum, reduction='none')
            # 应用样本权重
            weighted_loss = bce_loss * sample_weights.unsqueeze(1)
            loss = weighted_loss.mean()
        else:
            loss = F.binary_cross_entropy(probs, targets_cum, reduction='none').mean()
        
        return loss

class CoralEnhancedSuicideRiskClassifier(nn.Module):
    """
    基于CORAL的增强版自杀风险分类器
    - 使用累积序数回归处理有序分类问题
    - 去偏注意力机制
    - 多层次特征提取
    - 增强时间特征融合
    """
    
    def __init__(self, model_name='microsoft/deberta-v3-large', 
                 hidden_dim=512, time_feature_dim=12,
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
        
        # CORAL分类器 - 输出3个二分类logits
        self.coral_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 3)  # 3个二分类头
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化新增层的权重"""
        for module in [self.feature_fusion, self.coral_classifier]:
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
        
        # CORAL分类 - 输出3个二分类logits
        logits = self.coral_classifier(fused_features)
        return logits

def coral_predict(logits):
    """
    将CORAL模型的logits转换为最终类别预测
    
    Args:
        logits: 模型输出，形状为 (batch_size, 3)
        
    Returns:
        predictions: 预测类别，形状为 (batch_size,)
    """
    # 计算sigmoid概率
    probs = torch.sigmoid(logits)
    
    # 根据概率确定类别
    # y=0 如果 P(y>0) < 0.5
    # y=1 如果 P(y>0) >= 0.5 且 P(y>1) < 0.5
    # y=2 如果 P(y>1) >= 0.5 且 P(y>2) < 0.5
    # y=3 如果 P(y>2) >= 0.5
    preds = torch.zeros(probs.size(0), device=probs.device, dtype=torch.long)
    
    # P(y > 0) >= 0.5
    mask_1 = probs[:, 0] >= 0.5
    preds[mask_1] = 1
    
    # P(y > 1) >= 0.5
    mask_2 = (probs[:, 1] >= 0.5) & mask_1
    preds[mask_2] = 2
    
    # P(y > 2) >= 0.5
    mask_3 = (probs[:, 2] >= 0.5) & mask_2
    preds[mask_3] = 3
    
    return preds