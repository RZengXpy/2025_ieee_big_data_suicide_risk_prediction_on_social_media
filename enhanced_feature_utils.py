import ast
import re
import numpy as np
import pandas as pd

def _parse_single_to_dt(val):
    """把一个值（可能是 scalar/list/str）解析成单个 pandas.Timestamp（优先取列表的最后一个元素）。"""
    if val is None:
        return pd.NaT

    # 先处理 pandas Timestamp / numpy datetime64
    if isinstance(val, pd.Timestamp):
        return val
    try:
        if np.issubdtype(type(val), np.datetime64):
            return pd.to_datetime(val)
    except Exception:
        pass

    # 如果是 list/tuple/ndarray -> 递归取最后一个元素
    if isinstance(val, (list, tuple, np.ndarray)):
        if len(val) == 0:
            return pd.NaT
        return _parse_single_to_dt(val[-1])

    # 如果是字符串，可能是：
    #  - ISO 字符串："2021-05-01 05:57:38"
    #  - 字符串化列表："[Timestamp('2021-05-01 05:57:38'), ...]"
    if isinstance(val, str):
        s = val.strip()
        # 字符串化列表的情形：尝试 ast.literal_eval
        if s.startswith('[') and s.endswith(']'):
            try:
                parsed = ast.literal_eval(s)
                return _parse_single_to_dt(parsed[-1] if parsed else None)
            except Exception:
                # 可能格式像 " [Timestamp('...'), ...]" or "Timestamp('...')"
                # 尝试正则抽取 Timestamp('...') 里的时间
                m = re.search(r"Timestamp\(['\"]([^'\"]+)['\"]\)", s)
                if m:
                    try:
                        return pd.to_datetime(m.group(1))
                    except:
                        return pd.NaT
                # 否则最后退回到直接解析整个字符串
                try:
                    return pd.to_datetime(s)
                except:
                    return pd.NaT
        # 不是列表形式，直接尝试解析单个 iso 字符串
        try:
            return pd.to_datetime(s)
        except Exception:
            # 如果含 Timestamp('...') 格式
            m = re.search(r"Timestamp\(['\"]([^'\"]+)['\"]\)", s)
            if m:
                try:
                    return pd.to_datetime(m.group(1))
                except:
                    return pd.NaT
            return pd.NaT

    # 如果是数值，尝试按秒/ms/ns解析（贪婪尝试不同 unit）
    if isinstance(val, (int, float, np.integer, np.floating)):
        for unit in ['s', 'ms', 'us', 'ns']:
            try:
                dt = pd.to_datetime(val, unit=unit, errors='coerce')
                if not pd.isna(dt):
                    return dt
            except Exception:
                continue
        return pd.NaT

    # 其它类型无法解析
    return pd.NaT


def extract_enhanced_time_features(df):
    """
    增强版时间特征提取：
    - 自动寻找可用列，优先级： 'post_time_sequence' > 'post_timestamps' > 'post_times' > 'created_utc' > 'created_at'
    - 每行提取一个时间点（优先取序列最后一个时间），然后做周期编码
    - 新增月份和季节特征
    - 增强时间标记特征
    
    返回：numpy array (n_rows, 12)  -> [hour_s, hour_c, weekday_s, weekday_c, day_s, day_c, 
                                         month_s, month_c, is_night, is_weekend, is_holiday_season, is_working_hours]
    """
    candidates = ['post_time_sequence', 'post_timestamps', 'post_times', 'created_utc', 'created_at']
    # 找到 dataframe 中实际存在且非全空的列（按优先级）
    time_col = None
    for c in candidates:
        if c in df.columns:
            # 如果这一列不是全部为空则选它
            if df[c].notna().any():
                time_col = c
                break

    features = []
    if time_col is None:
        # 没有时间列，返回 zeros
        n = len(df)
        return np.zeros((n, 12), dtype=np.float32)

    # 逐行解析（为健壮性使用迭代，数据集若很大可后续向量化优化）
    for val in df[time_col].values:
        dt = _parse_single_to_dt(val)  # pandas.Timestamp 或 pd.NaT
        if pd.isna(dt):
            # 无有效时间，填 0 向量
            features.append([0.0] * 12)
            continue

        h = int(dt.hour)
        wd = int(dt.weekday())  # 0-6 (Monday=0)
        day = int(dt.day)
        month = int(dt.month)

        # 基础周期编码
        hour_s = np.sin(2 * np.pi * h / 24.0)
        hour_c = np.cos(2 * np.pi * h / 24.0)
        weekday_s = np.sin(2 * np.pi * wd / 7.0)
        weekday_c = np.cos(2 * np.pi * wd / 7.0)
        day_s = np.sin(2 * np.pi * day / 31.0)
        day_c = np.cos(2 * np.pi * day / 31.0)
        
        # 新增：月份周期编码
        month_s = np.sin(2 * np.pi * month / 12.0)
        month_c = np.cos(2 * np.pi * month / 12.0)

        # 基础时间标记
        is_night = 1.0 if (h >= 22 or h <= 6) else 0.0
        is_weekend = 1.0 if wd >= 5 else 0.0
        
        # 新增：节假日季节（假设12月-2月为节假日季节）
        is_holiday_season = 1.0 if month in [12, 1, 2] else 0.0
        
        # 新增：工作时间（周一到周五的9-18点）
        is_working_hours = 1.0 if (wd < 5 and 9 <= h <= 18) else 0.0

        features.append([
            hour_s, hour_c, weekday_s, weekday_c, day_s, day_c,
            month_s, month_c, is_night, is_weekend, is_holiday_season, is_working_hours
        ])

    return np.array(features, dtype=np.float32)


def extract_temporal_patterns(df):
    """
    提取更复杂的时间模式特征
    
    Args:
        df: 包含时间序列的DataFrame
        
    Returns:
        numpy array: 时间模式特征
    """
    candidates = ['post_time_sequence', 'post_timestamps', 'post_times']
    time_col = None
    
    for c in candidates:
        if c in df.columns and df[c].notna().any():
            time_col = c
            break
    
    if time_col is None:
        return np.zeros((len(df), 6), dtype=np.float32)
    
    pattern_features = []
    
    for val in df[time_col].values:
        if val is None or pd.isna(val):
            pattern_features.append([0.0] * 6)
            continue
            
        try:
            # 尝试解析时间序列
            if isinstance(val, str):
                if val.startswith('['):
                    time_list = ast.literal_eval(val)
                    if not time_list:
                        pattern_features.append([0.0] * 6)
                        continue
                    
                    # 转换为时间戳
                    timestamps = []
                    for t in time_list:
                        dt = _parse_single_to_dt(t)
                        if not pd.isna(dt):
                            timestamps.append(dt)
                    
                    if len(timestamps) < 2:
                        pattern_features.append([0.0] * 6)
                        continue
                    
                    # 计算时间模式特征
                    time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() / 3600 
                                 for i in range(1, len(timestamps))]  # 小时为单位
                    
                    # 特征：平均发帖间隔
                    avg_interval = np.mean(time_diffs) if time_diffs else 0.0
                    
                    # 特征：发帖间隔方差（规律性）
                    interval_variance = np.var(time_diffs) if len(time_diffs) > 1 else 0.0
                    
                    # 特征：发帖频率（帖子数/天）
                    total_span_days = (timestamps[-1] - timestamps[0]).total_seconds() / (24 * 3600)
                    post_frequency = len(timestamps) / max(total_span_days, 1)
                    
                    # 特征：夜间发帖比例
                    night_posts = sum(1 for ts in timestamps if ts.hour >= 22 or ts.hour <= 6)
                    night_ratio = night_posts / len(timestamps)
                    
                    # 特征：周末发帖比例
                    weekend_posts = sum(1 for ts in timestamps if ts.weekday() >= 5)
                    weekend_ratio = weekend_posts / len(timestamps)
                    
                    # 特征：发帖时间集中度（熵）
                    hours = [ts.hour for ts in timestamps]
                    hour_counts = np.bincount(hours, minlength=24)
                    hour_probs = hour_counts / len(timestamps)
                    # 计算熵（时间分布的均匀程度）
                    time_entropy = -np.sum(hour_probs * np.log(hour_probs + 1e-10))
                    
                    pattern_features.append([
                        min(avg_interval, 168),  # 限制在一周内
                        min(interval_variance, 1000),  # 限制方差
                        min(post_frequency, 10),  # 限制频率
                        night_ratio,
                        weekend_ratio,
                        time_entropy / np.log(24)  # 归一化熵
                    ])
                else:
                    pattern_features.append([0.0] * 6)
            else:
                pattern_features.append([0.0] * 6)
        except:
            pattern_features.append([0.0] * 6)
    
    return np.array(pattern_features, dtype=np.float32)


# 兼容性函数，保持原有接口
def extract_time_features(df):
    """保持向后兼容的时间特征提取函数"""
    return extract_enhanced_time_features(df)