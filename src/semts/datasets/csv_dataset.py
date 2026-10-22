"""滑动窗口数据集：支持 ETT / Weather / Electricity / Traffic / Exchange 等 CSV。

约定：首列（若有）为日期列（自动丢弃），其余列为数值通道。
标准化使用训练段统计（均值/方差），与 TS2Vec/CoST/SimMTM 协议一致。
窗口严格限制在各自 split 边界内（不跨边界泄漏）；split 长度不足时显式报错。
"""
import numpy as np
import pandas as pd


class SlidingWindowDataset:
    def __init__(self, path, L=336, date_col="date", train_ratio=0.6, val_ratio=0.2,
                 scale=True, max_channels=None):
        df = pd.read_csv(path)
        if date_col in df.columns:
            df = df.drop(columns=[date_col])
        df = df.select_dtypes(include=[np.number])
        if max_channels is not None:
            df = df.iloc[:, :max_channels]
        self.n_channels = df.shape[1]
        # ETT 等数据含 NaN（ETTm2 测试段尾部）：前向/后向填充 + 兜底置 0（标准协议）
        df = df.ffill().bfill().fillna(0.0)
        data = df.values.astype(np.float32)
        n = len(data)
        self.n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        self.mu = data[:self.n_train].mean(0, keepdims=True)
        self.sd = data[:self.n_train].std(0, keepdims=True) + 1e-5
        if scale:
            data = (data - self.mu) / self.sd
        self.data = data
        self.L = L
        # 各 split 的行范围 [a, b)；窗口起始索引 ∈ [a, b-L-extra]
        self.bounds = {
            "train": (0, self.n_train),
            "val": (self.n_train, self.n_train + n_val),
            "test": (self.n_train + n_val, n),
        }

    def _starts(self, split, extra=0, stride=1):
        """窗口起始索引数组；split 长度不足以构造窗口时抛错。"""
        a, b = self.bounds[split]
        hi = b - self.L - extra
        if hi < a:
            raise ValueError(
                f"split '{split}' 行数 {b - a} 不足以构造 L={self.L}"
                + (f"+H={extra}" if extra else "") + " 的窗口"
            )
        return np.arange(a, hi + 1, stride)

    def windows(self, split, stride=1):
        """返回 (N, L, C) numpy 数组。"""
        idx = self._starts(split, stride=stride)
        return self.data[idx[:, None] + np.arange(self.L)]

    def windows_with_target(self, split, H, stride=1):
        """返回 (x, y)：x: (N, L, C)，y: (N, H, C)（未来 H 步）。"""
        idx = self._starts(split, extra=H, stride=stride)
        x = self.data[idx[:, None] + np.arange(self.L)]
        y = self.data[idx[:, None] + np.arange(self.L, self.L + H)]
        return x, y
