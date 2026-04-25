# src/data_pipeline/dreamer/datamudule.py
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from typing import Optional

from src.data_pipeline.dreamer.tools.build_manifest import build_global_manifest
from src.data_pipeline.dreamer.mat_io import DreamerMatIO, WindowSpec
from src.data_pipeline.dreamer.dataset import DreamerDataset

def split_subject_independent_indices(
    manifest_df,
    test_sid: int,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    subject-independent 最小切分：
    - test: 指定 subject 全部样本
    - train/val: 其他 subjects 的样本按比例随机切分
    返回的是全局 manifest 的 row 索引数组
    """
    if "sid" not in manifest_df.columns:
        raise ValueError("manifest_df must contain column 'sid'")

    # 1) test: 指定 subject
    test_mask = manifest_df["sid"] == int(test_sid)
    test_indices = manifest_df.index[test_mask].to_numpy(dtype=np.int64)

    # 2) train_val_pool: 非 test subject
    pool_indices = manifest_df.index[~test_mask].to_numpy(dtype=np.int64)

    if len(test_indices) == 0:
        raise ValueError(f"No samples found for test_sid={test_sid}")
    if len(pool_indices) == 0:
        raise ValueError("No samples left for train/val after excluding test_sid")

    # 3) pool 内随机切 val
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(pool_indices)

    n_val = int(len(shuffled) * float(val_ratio))
    # 给个最小保护，避免 val 或 train 变空
    n_val = max(1, min(n_val, len(shuffled) - 1))

    val_indices = shuffled[:n_val]
    train_indices = shuffled[n_val:]

    return train_indices, val_indices, test_indices

class DreamerDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # 只取第一小步需要的3个关键字段
        self.dataset_name = cfg.dataset.dataset_name
        self.batch_size = cfg.train.batch_size
        self.num_workers = cfg.train.num_workers

        self.global_manifest = None
        self.io = None
        self.window_spec = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def summary(self) -> str:
        return (
            f"DreamerDataModule(dataset_name={self.dataset_name}, "
            f"batch_size={self.batch_size}, "
            f"num_workers={self.num_workers})"
        )
    
    def prepare_data(self) -> None:
        if self.dataset_name == "dreamer":
            print("prepare dreamer dataset...")

            self.io = DreamerMatIO(
                mat_path=self.cfg.dataset.mat_path,
                verify_compressed_data_integrity=False,
            )
            self.window_spec = WindowSpec(
                win_sec=self.cfg.dataset.win_sec,
                stride_sec=self.cfg.dataset.stride_sec,
            )

            self.global_manifest = build_global_manifest(
                io=self.io,
                window_spec=self.window_spec,
                is_force_rebuild=self.cfg.dataset.is_force_rebuild,
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        return None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit", "test"):
            train_idx, val_idx, test_idx = split_subject_independent_indices(
                manifest_df=self.global_manifest,
                test_sid=self.cfg.dataset.test_sid,   # 你在 yaml 里加这个字段
                val_ratio=self.cfg.dataset.val_ratio, # 你在 yaml 里加这个字段
                seed=self.cfg.dataset.seed,           # 你在 yaml 里加这个字段
            )
        # fit 时实例化 train/val
        if stage in (None, "fit"):
            self.train_dataset = DreamerDataset(
                self.io, 
                self.global_manifest, 
                self.window_spec,
                train_idx
            )
            self.val_dataset = DreamerDataset(
                self.io, 
                self.global_manifest, 
                self.window_spec, 
                val_idx
            )
        # test 时实例化 test
        if stage in (None, "test"):
            self.test_dataset = DreamerDataset(
                self.io, 
                self.global_manifest, 
                self.window_spec, 
                test_idx
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)