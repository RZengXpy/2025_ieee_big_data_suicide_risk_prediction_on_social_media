import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DebertaV2Model
import math
from enhanced_model import MultiHeadDebiasedAttention, MultiLevelFeatureExtractor, EnhancedTimeProjector

class CascadedClassifierStage(nn.Module):
    """
    级联分类器的单个阶段
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # 二分类
        )
        
    def forward(self, features):
        logits = self.classifier(features)
        return logits

class CascadedSuicideRiskClassifier(nn.Module):
    """
    级联自杀风险分类器
    Step1: 区分 0 vs (1,2,3)
    Step2: 区分 1 vs (2,3)
    Step3: 区分 2 vs 3
    
    推理时按照 cascade 流程走下去，逐层 refine。
    """
    
    def __init__(self, model_name='microsoft/deberta-v3-large', 
                 hidden_dim=512, time_feature_dim=12,
                 pattern_feature_dim=6, use_debiased_attention=False):
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
        
        # 三个级联阶段的分类器
        # Step1: 区分 0 vs (1,2,3)
        self.stage1_classifier = CascadedClassifierStage(hidden_dim, hidden_dim // 2)
        
        # Step2: 区分 1 vs (2,3)
        self.stage2_classifier = CascadedClassifierStage(hidden_dim, hidden_dim // 2)
        
        # Step3: 区分 2 vs 3
        self.stage3_classifier = CascadedClassifierStage(hidden_dim, hidden_dim // 2)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化新增层的权重"""
        for module in [self.feature_fusion, self.stage1_classifier, self.stage2_classifier, self.stage3_classifier]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        torch.nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            torch.nn.init.zeros_(layer.bias)
            elif hasattr(module, 'classifier'):  # CascadedClassifierStage
                for layer in module.classifier:
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
            print("无层次特征")
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
            print("无模式特征")
            pattern_emb = torch.zeros(B, time_emb.size(-1) // 2, device=time_emb.device)
            combined_features = torch.cat([text_features, time_emb, pattern_emb], dim=-1)
        
        # 特征融合
        fused_features = self.feature_fusion(combined_features)
        
        # 三个阶段的分类器
        stage1_logits = self.stage1_classifier(fused_features)  # 0 vs (1,2,3)
        stage2_logits = self.stage2_classifier(fused_features)  # 1 vs (2,3)
        stage3_logits = self.stage3_classifier(fused_features)  # 2 vs 3
        
        # 返回所有阶段的logits
        return stage1_logits, stage2_logits, stage3_logits

def cascaded_predict(stage1_logits, stage2_logits, stage3_logits):
    """
    级联分类器的预测函数
    
    级联流程：
    Step1: 区分 0 vs (1,2,3) - 如果是0类则直接返回，否则进入下一步
    Step2: 区分 1 vs (2,3) - 如果是1类则直接返回，否则进入下一步
    Step3: 区分 2 vs 3 - 返回2或3
    
    Args:
        stage1_logits: Step1的logits (batch_size, 1) - 区分 0 vs (1,2,3)
        stage2_logits: Step2的logits (batch_size, 1) - 区分 1 vs (2,3)
        stage3_logits: Step3的logits (batch_size, 1) - 区分 2 vs 3
        
    Returns:
        predictions: 预测类别 (batch_size,)
    """
    batch_size = stage1_logits.size(0)
    
    # 计算sigmoid概率
    prob_stage1 = torch.sigmoid(stage1_logits).squeeze(-1)  # 0 vs (1,2,3)
    prob_stage2 = torch.sigmoid(stage2_logits).squeeze(-1)  # 1 vs (2,3)
    prob_stage3 = torch.sigmoid(stage3_logits).squeeze(-1)  # 2 vs 3
    
    # 初始化预测结果
    predictions = torch.zeros(batch_size, dtype=torch.long, device=stage1_logits.device)
    
    # Step1: 区分 0 vs (1,2,3)
    # 如果P(0) >= 0.5 (即P(>0) < 0.5)，则预测为0
    is_class_0 = prob_stage1 < 0.5
    
    # Step2: 区分 1 vs (2,3) - 对非0类样本
    # 如果P(1|(1,2,3)) >= 0.5 (即P(>1) < 0.5)，则预测为1
    is_class_1 = prob_stage2 < 0.5
    
    # Step3: 区分 2 vs 3 - 对(2,3)类样本
    # 如果P(2|(2,3)) >= 0.5 (即P(>2) < 0.5)，则预测为2，否则为3
    is_class_2 = prob_stage3 < 0.5
    
    # 应用级联逻辑
    # 预测为0的样本
    predictions[is_class_0] = 0
    
    # 预测为1的样本：不属于0类 且 属于1类
    predictions[~is_class_0 & is_class_1] = 1
    
    # 预测为2的样本：不属于0类 且 不属于1类 且 属于2类
    predictions[~is_class_0 & ~is_class_1 & is_class_2] = 2
    
    # 预测为3的样本：不属于0类 且 不属于1类 且 不属于2类
    predictions[~is_class_0 & ~is_class_1 & ~is_class_2] = 3
    
    return predictions

class CascadedLoss(nn.Module):
    """
    级联分类器的损失函数
    """
    def __init__(self, weights=None):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=weights)
        
    def forward(self, stage1_logits, stage2_logits, stage3_logits, targets):
        """
        计算级联分类器的损失
        
        Args:
            stage1_logits: Step1的logits (batch_size, 1) - 区分 0 vs (1,2,3)
            stage2_logits: Step2的logits (batch_size, 1) - 区分 1 vs (2,3)
            stage3_logits: Step3的logits (batch_size, 1) - 区分 2 vs 3
            targets: 真实标签 (batch_size,)
        """
        # Step1: 0 vs (1,2,3)
        # 目标：0为负类，(1,2,3)为正类 (注意这里与直觉相反，因为我们要预测非0类)
        stage1_targets = (targets > 0).float()
        loss1 = self.bce_loss(stage1_logits.squeeze(-1), stage1_targets)
        
        # Step2: 1 vs (2,3) - 只对目标为1,2,3的样本计算
        mask_stage2 = targets > 0
        if mask_stage2.any():
            stage2_targets = (targets[mask_stage2] > 1).float()
            loss2 = self.bce_loss(stage2_logits[mask_stage2].squeeze(-1), stage2_targets)
        else:
            loss2 = 0.0
            
        # Step3: 2 vs 3 - 只对目标为2,3的样本计算
        mask_stage3 = targets > 1
        if mask_stage3.any():
            stage3_targets = (targets[mask_stage3] > 2).float()
            loss3 = self.bce_loss(stage3_logits[mask_stage3].squeeze(-1), stage3_targets)
        else:
            loss3 = 0.0
            
        # 总损失
        total_loss = loss1 + loss2 + loss3
        return total_loss