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


def extract_time_features(df):
    """
    兼容版时间特征提取：
    - 自动寻找可用列，优先级： 'post_time_sequence' > 'post_timestamps' > 'post_times' > 'created_utc' > 'created_at'
    - 每行提取一个时间点（优先取序列最后一个时间），然后做周期编码
    返回：numpy array (n_rows, 8)  -> [hour_s, hour_c, weekday_s, weekday_c, day_s, day_c, is_night, is_weekend]
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
        return np.zeros((n, 8), dtype=np.float32)

    # 逐行解析（为健壮性使用迭代，数据集若很大可后续向量化优化）
    for val in df[time_col].values:
        dt = _parse_single_to_dt(val)  # pandas.Timestamp 或 pd.NaT
        if pd.isna(dt):
            # 无有效时间，填 0 向量
            features.append([0.0]*8)
            continue

        h = int(dt.hour)
        wd = int(dt.weekday())  # 0-6
        day = int(dt.day)
        month = int(dt.month)

        hour_s = np.sin(2 * np.pi * h / 24.0)
        hour_c = np.cos(2 * np.pi * h / 24.0)
        weekday_s = np.sin(2 * np.pi * wd / 7.0)
        weekday_c = np.cos(2 * np.pi * wd / 7.0)
        day_s = np.sin(2 * np.pi * day / 31.0)
        day_c = np.cos(2 * np.pi * day / 31.0)
        is_night = 1.0 if (h >= 22 or h <= 6) else 0.0
        is_weekend = 1.0 if wd >= 5 else 0.0

        features.append([hour_s, hour_c, weekday_s, weekday_c, day_s, day_c, is_night, is_weekend])

    return np.array(features, dtype=np.float32)