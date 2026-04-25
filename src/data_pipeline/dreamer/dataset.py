from __future__ import annotations

from typing import Sequence, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data_pipeline.dreamer.mat_io import DreamerMatIO, WindowSpec


class DreamerDataset(Dataset):
    """
    最小可用版 Dreamer Dataset
    - 共享全局 manifest_df
    - 每个 split 只传自己的 indices
    - 先返回 EEG + 二分类标签（valence > threshold）
    """

    def __init__(
        self,
        io: DreamerMatIO,
        manifest_df: pd.DataFrame,
        window_spec: WindowSpec,
        indices: Sequence[int],
        label_threshold: float = 3.0,
        channels_first: bool = True,
        return_meta: bool = False,
    ) -> None:
        super().__init__()
        self.io = io
        self.manifest_df = manifest_df
        self.window_spec = window_spec
        self.indices = np.asarray(indices, dtype=np.int64)

        self.label_threshold = float(label_threshold)
        self.channels_first = channels_first
        self.return_meta = return_meta

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        row_idx = int(self.indices[i])
        row = self.manifest_df.iloc[row_idx]

        # 1) 取定位信息（manifest -> mat_io）
        sid = int(row["sid"])
        trial_id = int(row["trial_id"])
        window_id = int(row["window_id"])

        # 2) 用该行自己的窗口参数构造 WindowSpec（避免 cfg/manifest 不一致）

        # 3) 回源取窗口
        sample = self.io.get_window(
            sid=sid,
            trial_id=trial_id,
            window_id=window_id,
            spec=self.window_spec,
            channels_first=self.channels_first,
            dtype=np.float32,
        )

        # 4) 先只用 EEG 做输入
        # channels_first=True 时是 [C, T]
        x = torch.from_numpy(sample["eeg"]).float()

        # 5) valence 二分类标签
        valence = float(row["valence"])
        y = 1 if valence > self.label_threshold else 0
        y = torch.tensor(y, dtype=torch.long)

        if not self.return_meta:
            return x, y

        meta = {
            "row_id": int(row["row_id"]) if "row_id" in row else row_idx,
            "sample_id": str(row["sample_id"]) if "sample_id" in row else f"{sid}_{trial_id}_{window_id}",
            "sid": sid,
            "trial_id": trial_id,
            "window_id": window_id,
            "start_sec": float(row["start_sec"]),
            "end_sec": float(row["end_sec"]),
        }
        return x, y, meta