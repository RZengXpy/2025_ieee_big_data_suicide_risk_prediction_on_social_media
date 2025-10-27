import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import DebertaV2Tokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
import swanlab
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# 导入增强版模块
from enhanced_data_deal import EnhancedSuicideDataset, SuicideDataset
from enhanced_model import EnhancedSuicideRiskClassifier, SuicideRiskClassifier
from enhanced_feature_utils import extract_enhanced_time_features, extract_temporal_patterns
# 导入CORAL模块
from coral_model import CoralEnhancedSuicideRiskClassifier, CoralLoss, coral_predict
# 导入级联分类器模块
from cascaded_model import CascadedSuicideRiskClassifier, CascadedLoss, cascaded_predict

# 环境配置
# 可选 0 1 2 3
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(42)
# np.random.seed(42)
# 62
# 76
# 42

MODEL_NAME = '/home/zr/.cache/modelscope/hub/models/microsoft/deberta-v3-large'
save_path = '/home/zr/bigdata/card_model2/'
file_path = '/home/zr/bigdata/dataset/train.pkl'

def prepare_user_stratified_folds(df, n_splits=2):
    """
    按用户分组+标签分层，生成交叉验证折
    """
    if 'user_id' not in df.columns:
        raise ValueError("数据中缺少 user_id 列，无法按用户划分。")
    if 'suicide_risk' not in df.columns:
        raise ValueError("数据中缺少 suicide_risk 列，无法进行分层。")

    # 统一每个用户的标签（取众数）
    user_labels_map = {}
    for uid, group in df.groupby('user_id')['suicide_risk']:
        label_counts = Counter(group)
        main_label, count = label_counts.most_common(1)[0]
        if len(label_counts) > 1:
            print(f"⚠️ 用户 {uid} 存在多标签 {dict(label_counts)}，使用多数标签 {main_label}")
        user_labels_map[uid] = main_label

    user_ids = list(user_labels_map.keys())
    labels = [user_labels_map[uid] for uid in user_ids]

    # 用户级分层交叉验证
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = []
    for train_idx, val_idx in sgkf.split(user_ids, labels, groups=user_ids):
        train_ids = [user_ids[i] for i in train_idx]
        val_ids = [user_ids[i] for i in val_idx]
        train_df = df[df['user_id'].isin(train_ids)].reset_index(drop=True)
        val_df = df[df['user_id'].isin(val_ids)].reset_index(drop=True)
        folds.append((train_df, val_df))
    return folds

def train_epoch(model, dataloader, optimizer, criterion, scheduler, use_enhanced=True, use_coral=False, use_cascaded=False):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        try:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            time_features = batch['time_features'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            
            if use_enhanced:
                # 使用增强模型
                pattern_features = batch['pattern_features'].to(device)
                outputs = model(input_ids, attention_mask, time_features, pattern_features)
            else:
                # 使用原版模型
                outputs = model(input_ids, attention_mask, time_features)
            
            # 根据使用的模型类型调整损失计算
            if use_coral:
                loss = criterion(outputs, labels)
            elif use_cascaded:
                stage1_logits, stage2_logits, stage3_logits = outputs
                loss = criterion(stage1_logits, stage2_logits, stage3_logits, labels)
            else:
                loss = criterion(outputs, labels)
                
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
            
            if step % 100 == 0:
                print(f"✅ Step {step} 完成, loss: {loss.item():.4f}")
        except Exception as e:
            print(f"❌ Step {step} 出错: {e}")
    
    return total_loss / len(dataloader)

def eval_epoch(model, criterion, dataloader, use_enhanced=True, use_coral=False, use_cascaded=False, num_classes=4):
    """验证一个epoch，返回多种评估指标"""
    model.eval()
    preds, trues, probs = [], [], []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                time_features = batch['time_features'].to(device)
                labels = batch['label'].to(device)

                if use_enhanced:
                    pattern_features = batch['pattern_features'].to(device)
                    outputs = model(input_ids, attention_mask, time_features, pattern_features)
                else:
                    outputs = model(input_ids, attention_mask, time_features)
                
                # 根据使用的模型类型调整损失计算和预测
                if use_coral:
                    loss = criterion(outputs, labels)
                    # 使用CORAL预测函数
                    pred = coral_predict(outputs)
                elif use_cascaded:
                    stage1_logits, stage2_logits, stage3_logits = outputs
                    loss = criterion(stage1_logits, stage2_logits, stage3_logits, labels)
                    # 使用级联预测函数
                    pred = cascaded_predict(stage1_logits, stage2_logits, stage3_logits)
                else:
                    loss = criterion(outputs, labels)
                    pred = outputs.argmax(dim=1)
                    
                total_loss += loss.item()
                
                preds.extend(pred.cpu().numpy())
                trues.extend(labels.cpu().numpy())
                
                # 保存概率用于计算AUC
                # 对于级联分类器，我们需要特殊处理
                if use_cascaded:
                    # 级联分类器使用预测函数进行预测，但我们仍需要概率来计算AUC
                    # 这里我们使用sigmoid来计算每个阶段的概率
                    prob_stage1 = torch.sigmoid(stage1_logits).squeeze(-1)  # 0 vs (1,2,3)
                    prob_stage2 = torch.sigmoid(stage2_logits).squeeze(-1)  # 1 vs (2,3)
                    prob_stage3 = torch.sigmoid(stage3_logits).squeeze(-1)  # 2 vs 3
                    
                    # 构造每个类别的概率
                    batch_size = len(pred)
                    batch_probs = np.zeros((batch_size, 4))
                    
                    # 类别0的概率：P(0) = 1 - P(>0) = 1 - prob_stage1
                    batch_probs[:, 0] = 1 - prob_stage1.cpu().numpy()
                    
                    # 类别1的概率：P(1) = P(>0) * (1 - P(>1)) = prob_stage1 * (1 - prob_stage2)
                    batch_probs[:, 1] = prob_stage1.cpu().numpy() * (1 - prob_stage2.cpu().numpy())
                    
                    # 类别2的概率：P(2) = P(>0) * P(>1) * (1 - P(>2)) = prob_stage1 * prob_stage2 * (1 - prob_stage3)
                    batch_probs[:, 2] = prob_stage1.cpu().numpy() * prob_stage2.cpu().numpy() * (1 - prob_stage3.cpu().numpy())
                    
                    # 类别3的概率：P(3) = P(>0) * P(>1) * P(>2) = prob_stage1 * prob_stage2 * prob_stage3
                    batch_probs[:, 3] = prob_stage1.cpu().numpy() * prob_stage2.cpu().numpy() * prob_stage3.cpu().numpy()
                    
                    probs.extend(batch_probs)
                else:
                    # 普通模型和CORAL模型输出单个张量
                    prob = torch.softmax(outputs, dim=1)
                    probs.extend(prob.cpu().numpy())
                
            except Exception as e:
                print(f"❌ 验证批次出错: {e}")

    # 计算各种评估指标
    # F1分数
    f1_weighted = f1_score(trues, preds, average='weighted')
    f1_macro = f1_score(trues, preds, average='macro')
    f1_micro = f1_score(trues, preds, average='micro')
    
    # 精确率和召回率
    precision_per_class = precision_score(trues, preds, average=None, labels=range(num_classes))
    recall_per_class = recall_score(trues, preds, average=None, labels=range(num_classes))
    precision_macro = precision_score(trues, preds, average='macro')
    recall_macro = recall_score(trues, preds, average='macro')
    precision_micro = precision_score(trues, preds, average='micro')
    recall_micro = recall_score(trues, preds, average='micro')
    
    # AUC (需要处理多分类)
    trues_bin = label_binarize(trues, classes=range(num_classes))
    if num_classes == 2:
        auc_roc = roc_auc_score(trues_bin, np.array(probs)[:, 1])
    else:
        try:
            auc_roc = roc_auc_score(trues_bin, probs, multi_class='ovr')
        except ValueError:
            auc_roc = 0.0  # 当只有一个类别的时候会出现错误
    
    # 混淆矩阵
    cm = confusion_matrix(trues, preds)
    
    metrics = {
        'loss': total_loss / len(dataloader),
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'auc_roc': auc_roc,
        'confusion_matrix': cm,
        'predictions': preds,
        'targets': trues,
        'probabilities': probs
    }
    
    # 最后清理
    torch.cuda.empty_cache()
    
    return metrics

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """绘制混淆矩阵并返回图像"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    return plt.gcf()

def main_enhanced(file_path, use_enhanced_model=True, use_enhanced_features=True, use_coral=False, use_cascaded=False):
    """
    增强版主训练函数
    
    Args:
        file_path: 数据文件路径
        use_enhanced_model: 是否使用增强模型
        use_enhanced_features: 是否使用增强时间特征
        use_coral: 是否使用CORAL方法
        use_cascaded: 是否使用级联分类器
    """
    
    model_suffix = "enhanced" if use_enhanced_model else "standard"
    feature_suffix = "enhanced_features" if use_enhanced_features else "basic_features"
    coral_suffix = "_coral" if use_coral else ""
    cascaded_suffix = "_cascaded" if use_cascaded else ""
    
    # 初始化SwanLab
    swanlab.init(
        project="suicide-risk-enhanced",
        run_name=f"deberta_{model_suffix}_{feature_suffix}{coral_suffix}{cascaded_suffix}_5fold",
        config={
            "model_name": MODEL_NAME,
            "batch_size": 2, 
            "learning_rate": 2e-6,
            "max_len": 256,  
            "max_posts": 5,
            "hidden_dim": 512,
            "epochs": 15,
            "n_folds": 5,
            "weights": [1, 1.0, 1.0, 4.0],
            "use_enhanced_model": use_enhanced_model,
            "use_enhanced_features": use_enhanced_features,
            "use_debiased_attention": True,
            "use_coral": use_coral,
            "use_cascaded": use_cascaded,
            "time_feature_dim": 12 if use_enhanced_features else 8,
            "pattern_feature_dim": 6 if use_enhanced_features else 0
        }
    )
    
    config = swanlab.config
    
    # 读取数据
    df = pd.read_pickle(file_path)
    print(f"📊 数据加载完成，总样本数: {len(df)}")
    
    # 提取时间特征
    print("🕐 提取时间特征中...")
    if use_enhanced_features:
        time_features = extract_enhanced_time_features(df)  # 12维增强特征
        pattern_features = extract_temporal_patterns(df)    # 6维时间模式特征
        print(f"✅ 增强时间特征提取完成，基础特征: {time_features.shape}, 模式特征: {pattern_features.shape}")
    else:
        # 使用原版特征提取（保持兼容）
        from feature_utils import extract_time_features
        time_features = extract_time_features(df)  # 8维基础特征
        pattern_features = None
        print(f"✅ 基础时间特征提取完成，特征维度: {time_features.shape}")
    
    # 生成用户级分层交叉验证折
    folds = prepare_user_stratified_folds(df, n_splits=config.n_folds)
    
    all_fold_f1s = []
    
    for fold_idx, (train_df, val_df) in enumerate(folds, 1):
        print(f"\n📂 Fold {fold_idx}/{len(folds)} - 训练样本: {len(train_df)}, 验证样本: {len(val_df)}")
        
        # 提取对应折的时间特征
        train_indices = train_df.index.tolist()
        val_indices = val_df.index.tolist()
        train_time_feats = time_features[train_indices]
        val_time_feats = time_features[val_indices]
        
        if use_enhanced_features and pattern_features is not None:
            train_pattern_feats = pattern_features[train_indices]
            val_pattern_feats = pattern_features[val_indices]
        else:
            train_pattern_feats = val_pattern_feats = None
        
        # 初始化tokenizer和数据集
        tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
        
        if use_enhanced_features:
            # 使用增强数据集
            train_dataset = EnhancedSuicideDataset(
                train_df, tokenizer, 
                max_len=config.max_len, 
                max_posts=config.max_posts,
                time_features=train_time_feats,
                pattern_features=train_pattern_feats
            )
            val_dataset = EnhancedSuicideDataset(
                val_df, tokenizer, 
                max_len=config.max_len, 
                max_posts=config.max_posts,
                time_features=val_time_feats,
                pattern_features=val_pattern_feats
            )
        else:
            # 使用原版数据集
            train_dataset = SuicideDataset(
                train_df, tokenizer, 
                max_len=config.max_len, 
                max_posts=config.max_posts,
                time_features=train_time_feats
            )
            val_dataset = SuicideDataset(
                val_df, tokenizer, 
                max_len=config.max_len, 
                max_posts=config.max_posts,
                time_features=val_time_feats
            )

        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True
        )
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

        # 初始化模型
        if use_coral and use_enhanced_model:
            # 使用CORAL增强模型
            model = CoralEnhancedSuicideRiskClassifier(
                model_name=config.model_name, 
                hidden_dim=config.hidden_dim,
                time_feature_dim=config.time_feature_dim,
                pattern_feature_dim=config.pattern_feature_dim,
                use_debiased_attention=config.use_debiased_attention
            ).to(device)
        elif use_cascaded and use_enhanced_model:
            print("使用级联增强模型")
            # 使用级联增强模型
            model = CascadedSuicideRiskClassifier(
                model_name=config.model_name, 
                hidden_dim=config.hidden_dim,
                time_feature_dim=config.time_feature_dim,
                pattern_feature_dim=config.pattern_feature_dim,
                use_debiased_attention=config.use_debiased_attention
            ).to(device)
        elif use_enhanced_model:
            # 使用普通增强模型
            model = EnhancedSuicideRiskClassifier(
                model_name=config.model_name, 
                hidden_dim=config.hidden_dim,
                time_feature_dim=config.time_feature_dim,
                pattern_feature_dim=config.pattern_feature_dim,
                use_debiased_attention=config.use_debiased_attention
            ).to(device)
        else:
            # 使用原版模型
            model = SuicideRiskClassifier(
                model_name=config.model_name, 
                hidden_dim=config.hidden_dim,
                time_feature_dim=config.time_feature_dim
            ).to(device)

        # 损失函数
        weights = torch.tensor(config.weights, dtype=torch.float).to(device)
        if use_coral:
            criterion = CoralLoss(weight=weights)
        elif use_cascaded:
            criterion = CascadedLoss(weights=None)  # 级联分类器有自己的损失计算方式
        else:
            criterion = nn.CrossEntropyLoss(weight=weights)
        
        # 使用不同的学习率for不同的模块
        if use_enhanced_model:
            bert_params = list(model.bert.parameters())
            other_params = [p for p in model.parameters() if not any(p is bp for bp in bert_params)]
            
            optimizer = torch.optim.AdamW([
                {'params': bert_params, 'lr': config.learning_rate},
                {'params': other_params, 'lr': config.learning_rate * 2}  # 新增层使用更高学习率
            ])
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        # 学习率调度器
        num_training_steps = len(train_loader) * config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps
        )

        # 训练参数
        best_f1 = 0.45
        patience = 10
        patience_counter = 0
        
        # 存储所有折的评估指标
        all_fold_metrics = {
            'f1_weighted': [],
            'f1_macro': [],
            'f1_micro': [],
            'precision_macro': [],
            'recall_macro': [],
            'precision_micro': [],
            'recall_micro': [],
            'auc_roc': []
        }

        # 开始训练
        for epoch in range(config.epochs):
            print(f"\n🚀 [Fold {fold_idx}] Epoch {epoch+1}/{config.epochs}")
            
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, scheduler, 
                use_enhanced=use_enhanced_features, use_coral=use_coral, use_cascaded=use_cascaded
            )
            
            # 使用新的评估函数
            val_metrics = eval_epoch(
                model, criterion, val_loader, 
                use_enhanced=use_enhanced_features, use_coral=use_coral, use_cascaded=use_cascaded
            )
            
            val_loss = val_metrics['loss']
            val_f1_weighted = val_metrics['f1_weighted']
            val_f1_macro = val_metrics['f1_macro']
            val_f1_micro = val_metrics['f1_micro']
            val_precision_macro = val_metrics['precision_macro']
            val_recall_macro = val_metrics['recall_macro']
            val_precision_micro = val_metrics['precision_micro']
            val_recall_micro = val_metrics['recall_micro']
            val_auc_roc = val_metrics['auc_roc']
            val_precision_per_class = val_metrics['precision_per_class']
            val_recall_per_class = val_metrics['recall_per_class']
            cm = val_metrics['confusion_matrix']

            # 记录日志
            swanlab.log({
                f"fold_{fold_idx}/epoch": epoch + 1,
                f"fold_{fold_idx}/train_loss": train_loss,
                f"fold_{fold_idx}/val_loss": val_loss,
                f"fold_{fold_idx}/val_f1_weighted": val_f1_weighted,
                f"fold_{fold_idx}/val_f1_macro": val_f1_macro,
                f"fold_{fold_idx}/val_f1_micro": val_f1_micro,
                f"fold_{fold_idx}/val_precision_macro": val_precision_macro,
                f"fold_{fold_idx}/val_recall_macro": val_recall_macro,
                f"fold_{fold_idx}/val_precision_micro": val_precision_micro,
                f"fold_{fold_idx}/val_recall_micro": val_recall_micro,
                f"fold_{fold_idx}/val_auc_roc": val_auc_roc,
                f"fold_{fold_idx}/learning_rate": scheduler.get_last_lr()[0]
            })

            # 记录每个类别的精确率和召回率
            class_names = ['Risk_0', 'Risk_1', 'Risk_2', 'Risk_3']
            for i, class_name in enumerate(class_names):
                swanlab.log({
                    f"fold_{fold_idx}/precision_{class_name}": val_precision_per_class[i],
                    f"fold_{fold_idx}/recall_{class_name}": val_recall_per_class[i]
                })

            print(f"[Fold {fold_idx}] Epoch {epoch+1} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  F1 Scores - Weighted: {val_f1_weighted:.4f}, Macro: {val_f1_macro:.4f}, Micro: {val_f1_micro:.4f}")
            print(f"  Precision (Macro): {val_precision_macro:.4f}, Recall (Macro): {val_recall_macro:.4f}")
            print(f"  Precision (Micro): {val_precision_micro:.4f}, Recall (Micro): {val_recall_micro:.4f}")
            print(f"  AUC-ROC: {val_auc_roc:.4f}")
            
            # 打印每个类别的精确率和召回率
            print("  Per-class Metrics:")
            for i, class_name in enumerate(class_names):
                print(f"    {class_name} - Precision: {val_precision_per_class[i]:.4f}, Recall: {val_recall_per_class[i]:.4f}")
            
            # 保存最佳模型 (基于weighted F1)
            if val_f1_weighted > best_f1:
                patience_counter = 0
                best_f1 = val_f1_weighted
                model_name = f'enhanced_fold{fold_idx}_f1{val_f1_weighted:.4f}_best.pt' if use_enhanced_model else f'standard_fold{fold_idx}_f1{val_f1_weighted:.4f}_best.pt'
                if use_coral:
                    model_name = f'coral_{model_name}'
                elif use_cascaded:
                    model_name = f'cascaded_{model_name}'
                torch.save(
                    model.state_dict(), 
                    os.path.join(save_path, model_name)
                )
                print(f"💾 保存最佳模型: F1 = {val_f1_weighted:.4f}")
                
                # 生成并记录混淆矩阵图像
                plt.clf()  # 清除之前的图像
                cm_fig = plot_confusion_matrix(cm, class_names, f'Confusion Matrix - Fold {fold_idx}, Epoch {epoch+1}')
                swanlab.log({f"fold_{fold_idx}/confusion_matrix": swanlab.Image(cm_fig)})
                plt.close(cm_fig)  # 关闭图像以释放内存
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"⛔ Fold {fold_idx} 早停，最佳F1: {best_f1:.4f}")
                    break
        
        # 在每个fold结束后，记录最终的评估指标
        all_fold_metrics['f1_weighted'].append(val_f1_weighted)
        all_fold_metrics['f1_macro'].append(val_f1_macro)
        all_fold_metrics['f1_micro'].append(val_f1_micro)
        all_fold_metrics['precision_macro'].append(val_precision_macro)
        all_fold_metrics['recall_macro'].append(val_recall_macro)
        all_fold_metrics['precision_micro'].append(val_precision_micro)
        all_fold_metrics['recall_micro'].append(val_recall_micro)
        all_fold_metrics['auc_roc'].append(val_auc_roc)
        
        print(f"✅ Fold {fold_idx} 完成，最佳F1-weighted: {best_f1:.4f}")
    
    # 计算平均性能
    mean_metrics = {}
    std_metrics = {}
    for metric_name, values in all_fold_metrics.items():
        mean_metrics[metric_name] = np.mean(values)
        std_metrics[metric_name] = np.std(values)
    
    # 记录总体统计信息
    swanlab.log({
        "overall/mean_f1_weighted": mean_metrics['f1_weighted'],
        "overall/std_f1_weighted": std_metrics['f1_weighted'],
        "overall/mean_f1_macro": mean_metrics['f1_macro'],
        "overall/std_f1_macro": std_metrics['f1_macro'],
        "overall/mean_f1_micro": mean_metrics['f1_micro'],
        "overall/std_f1_micro": std_metrics['f1_micro'],
        "overall/mean_precision_macro": mean_metrics['precision_macro'],
        "overall/std_precision_macro": std_metrics['precision_macro'],
        "overall/mean_recall_macro": mean_metrics['recall_macro'],
        "overall/std_recall_macro": std_metrics['recall_macro'],
        "overall/mean_precision_micro": mean_metrics['precision_micro'],
        "overall/std_precision_micro": std_metrics['precision_micro'],
        "overall/mean_recall_micro": mean_metrics['recall_micro'],
        "overall/std_recall_micro": std_metrics['recall_micro'],
        "overall/mean_auc_roc": mean_metrics['auc_roc'],
        "overall/std_auc_roc": std_metrics['auc_roc']
    })
    
    print(f"\n📊 5折交叉验证结果:")
    print(f"模型类型: {'增强模型' if use_enhanced_model else '标准模型'}")
    print(f"特征类型: {'增强特征' if use_enhanced_features else '基础特征'}")
    print(f"CORAL方法: {'是' if use_coral else '否'}")
    print(f"级联分类器: {'是' if use_cascaded else '否'}")
    print(f"各折F1-weighted分数: {[f'{f1:.4f}' for f1 in all_fold_metrics['f1_weighted']]}")
    print(f"平均F1-weighted: {mean_metrics['f1_weighted']:.4f} ± {std_metrics['f1_weighted']:.4f}")
    print(f"平均F1-macro: {mean_metrics['f1_macro']:.4f} ± {std_metrics['f1_macro']:.4f}")
    print(f"平均F1-micro: {mean_metrics['f1_micro']:.4f} ± {std_metrics['f1_micro']:.4f}")
    print(f"平均Precision-macro: {mean_metrics['precision_macro']:.4f} ± {std_metrics['precision_macro']:.4f}")
    print(f"平均Recall-macro: {mean_metrics['recall_macro']:.4f} ± {std_metrics['recall_macro']:.4f}")
    print(f"平均Precision-micro: {mean_metrics['precision_micro']:.4f} ± {std_metrics['precision_micro']:.4f}")
    print(f"平均Recall-micro: {mean_metrics['recall_micro']:.4f} ± {std_metrics['recall_micro']:.4f}")
    print(f"平均AUC-ROC: {mean_metrics['auc_roc']:.4f} ± {std_metrics['auc_roc']:.4f}")
    
    swanlab.finish()

if __name__ == '__main__':
    
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    # 运行不同配置的实验
 #   print("🚀 实验1: 增强模型 + 增强特征")
 #   main_enhanced(file_path, use_enhanced_model=True, use_enhanced_features=True, use_coral=False, use_cascaded=False)
    
#    print("🚀 实验2: 增强模型 + 增强特征 + CORAL")
#    main_enhanced(file_path, use_enhanced_model=True, use_enhanced_features=True, use_coral=True, use_cascaded=False)
    
    print("🚀 实验3: 增强模型 + 增强特征 + 级联分类器")
    main_enhanced(file_path, use_enhanced_model=True, use_enhanced_features=True, use_coral=False, use_cascaded=True)
    
    # print("\n🚀 实验4: 标准模型 + 基础特征")
    # main_enhanced(file_path, use_enhanced_model=False, use_enhanced_features=False)
    
    # print("\n🚀 实验5: 增强模型 + 基础特征")
    # main_enhanced(file_path, use_enhanced_model=True, use_enhanced_features=False)


