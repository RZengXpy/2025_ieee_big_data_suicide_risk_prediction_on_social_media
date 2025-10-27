import torch
from torch.utils.data import Dataset
import ast
import numpy as np

class EnhancedSuicideDataset(Dataset):
    """
    增强版数据集，支持多种特征
    """
    def __init__(self, dataframe, tokenizer, max_len=128, max_posts=5, 
                 with_labels=True, time_features=None, pattern_features=None):
        """
        增强版数据集类，支持多帖子序列 + 多种时间特征
        
        Args:
            dataframe: 数据DataFrame
            tokenizer: 分词器
            max_len: 每个帖子的最大长度
            max_posts: 最大帖子数量
            with_labels: 是否包含标签
            time_features: 基础时间特征 (n_samples, 12)
            pattern_features: 时间模式特征 (n_samples, 6)
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_posts = max_posts
        self.data = dataframe
        self.with_labels = with_labels
        
        # 处理基础时间特征
        if time_features is not None:
            self.time_features = torch.tensor(time_features, dtype=torch.float32)
        else:
            # 如果没有时间特征，创建零向量
            self.time_features = torch.zeros(len(dataframe), 12, dtype=torch.float32)
        
        # 处理时间模式特征
        if pattern_features is not None:
            self.pattern_features = torch.tensor(pattern_features, dtype=torch.float32)
        else:
            # 如果没有模式特征，创建零向量
            self.pattern_features = torch.zeros(len(dataframe), 6, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def _safe_literal_eval(self, val):
        """安全地解析字符串为Python对象"""
        if isinstance(val, list):
            return val
        elif isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                # 如果解析失败，尝试其他方法
                # 移除可能的多余引号或转义字符
                cleaned = val.strip()
                if cleaned.startswith('"') and cleaned.endswith('"'):
                    cleaned = cleaned[1:-1]
                if cleaned.startswith("'") and cleaned.endswith("'"):
                    cleaned = cleaned[1:-1]
                try:
                    return ast.literal_eval(cleaned)
                except (ValueError, SyntaxError):
                    # 如果仍然失败，返回空列表
                    print(f"Warning: Could not parse post sequence: {val[:100]}...")
                    return []
        else:
            return [str(val)] if val is not None else []

    def __getitem__(self, index):
        # 处理帖子序列
        posts = self._safe_literal_eval(self.data.iloc[index]['post_sequence'])
        
        # 截断或补齐
        if len(posts) > self.max_posts:
            posts = posts[-self.max_posts:]
        else:
            posts = [''] * (self.max_posts - len(posts)) + posts

        # 编码文本
        encoded = self.tokenizer(
            posts, 
            padding='max_length', 
            truncation=True,
            max_length=self.max_len, 
            return_tensors='pt'
        )

        item = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'time_features': self.time_features[index],      # 基础时间特征
            'pattern_features': self.pattern_features[index]  # 时间模式特征
        }

        if self.with_labels:
            label = int(self.data.iloc[index]['suicide_risk'])
            item['label'] = torch.tensor(label, dtype=torch.long)

        return item


# 保持向后兼容的原版数据集
class SuicideDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128, max_posts=5, 
                 with_labels=True, time_features=None):
        """
        原版数据集类（向后兼容）
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_posts = max_posts
        self.data = dataframe
        self.with_labels = with_labels
        
        # 处理时间特征
        if time_features is not None:
            self.time_features = torch.tensor(time_features, dtype=torch.float32)
        else:
            # 如果没有时间特征，创建零向量 (保持原来的8维)
            self.time_features = torch.zeros(len(dataframe), 8, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def _safe_literal_eval(self, val):
        """安全地解析字符串为Python对象"""
        if isinstance(val, list):
            return val
        elif isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                # 如果解析失败，尝试其他方法
                # 移除可能的多余引号或转义字符
                cleaned = val.strip()
                if cleaned.startswith('"') and cleaned.endswith('"'):
                    cleaned = cleaned[1:-1]
                if cleaned.startswith("'") and cleaned.endswith("'"):
                    cleaned = cleaned[1:-1]
                try:
                    return ast.literal_eval(cleaned)
                except (ValueError, SyntaxError):
                    # 如果仍然失败，返回空列表
                    print(f"Warning: Could not parse post sequence: {val[:100]}...")
                    return []
        else:
            return [str(val)] if val is not None else []

    def __getitem__(self, index):
        # 处理帖子序列
        posts = self._safe_literal_eval(self.data.iloc[index]['post_sequence'])
        
        # 截断或补齐
        if len(posts) > self.max_posts:
            posts = posts[-self.max_posts:]
        else:
            posts = [''] * (self.max_posts - len(posts)) + posts

        # 编码文本
        encoded = self.tokenizer(
            posts, 
            padding='max_length', 
            truncation=True,
            max_length=self.max_len, 
            return_tensors='pt'
        )

        item = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'time_features': self.time_features[index]  # 添加时间特征
        }

        if self.with_labels:
            label = int(self.data.iloc[index]['suicide_risk'])
            item['label'] = torch.tensor(label, dtype=torch.long)

        return item