# src/data_pipeline/dreamer/datamodule.py  # [MOD-0] 路径注释拼写统一
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

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

    test_mask = manifest_df["sid"] == int(test_sid)
    test_indices = manifest_df.index[test_mask].to_numpy(dtype=np.int64)

    pool_indices = manifest_df.index[~test_mask].to_numpy(dtype=np.int64)

    if len(test_indices) == 0:
        raise ValueError(f"No samples found for test_sid={test_sid}")
    if len(pool_indices) == 0:
        raise ValueError("No samples left for train/val after excluding test_sid")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(pool_indices)

    n_val = int(len(shuffled) * float(val_ratio))
    n_val = max(1, min(n_val, len(shuffled) - 1))

    val_indices = shuffled[:n_val]
    train_indices = shuffled[n_val:]
    return train_indices, val_indices, test_indices


class DreamerDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.dataset_name = cfg.dataset.dataset_name
        self.batch_size = cfg.train.batch_size
        self.num_workers = cfg.train.num_workers

        self.global_manifest = None
        self.io = None
        self.window_spec = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # [MOD-1] 索引缓存，避免重复切分
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None

    def summary(self) -> str:
        return (
            f"DreamerDataModule(dataset_name={self.dataset_name}, "
            f"batch_size={self.batch_size}, "
            f"num_workers={self.num_workers})"
        )

    def prepare_data(self) -> None:
        # [MOD-2] prepare_data 只做“可重复准备动作”，不依赖实例内存状态
        # 这里可选：仅触发一次 manifest 构建（文件缓存层面）
        if self.dataset_name != "dreamer":
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        io = DreamerMatIO(
            mat_path=self.cfg.dataset.mat_path,
            verify_compressed_data_integrity=False,
        )
        window_spec = WindowSpec(
            win_sec=self.cfg.dataset.win_sec,
            stride_sec=self.cfg.dataset.stride_sec,
        )
        _ = build_global_manifest(
            io=io,
            window_spec=window_spec,
            is_force_rebuild=self.cfg.dataset.is_force_rebuild,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset_name != "dreamer":
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        # [MOD-3] 在 setup 里初始化运行时对象（多进程更稳）
        if self.io is None:
            self.io = DreamerMatIO(
                mat_path=self.cfg.dataset.mat_path,
                verify_compressed_data_integrity=False,
            )
        if self.window_spec is None:
            self.window_spec = WindowSpec(
                win_sec=self.cfg.dataset.win_sec,
                stride_sec=self.cfg.dataset.stride_sec,
            )
        if self.global_manifest is None:
            self.global_manifest = build_global_manifest(
                io=self.io,
                window_spec=self.window_spec,
                is_force_rebuild=self.cfg.dataset.is_force_rebuild,
            )

        # [MOD-4] 防御式检查，报错更清晰
        if self.global_manifest is None or len(self.global_manifest) == 0:
            raise RuntimeError("global_manifest is empty. Please check data source/build_manifest.")
        required_cols = {"sid"}
        missing = required_cols - set(self.global_manifest.columns)
        if missing:
            raise RuntimeError(f"global_manifest missing required columns: {missing}")

        # [MOD-5] 包含 validate 阶段，避免 trainer.validate() 时 val_dataset 为空
        if stage in (None, "fit", "validate", "test"):
            if self.train_idx is None or self.val_idx is None or self.test_idx is None:
                self.train_idx, self.val_idx, self.test_idx = split_subject_independent_indices(
                    manifest_df=self.global_manifest,
                    test_sid=self.cfg.dataset.test_sid,
                    val_ratio=self.cfg.dataset.val_ratio,
                    seed=self.cfg.dataset.seed,
                )

        if stage in (None, "fit"):
            self.train_dataset = DreamerDataset(
                self.io,
                self.global_manifest,
                self.window_spec,
                self.train_idx,
            )
            self.val_dataset = DreamerDataset(
                self.io,
                self.global_manifest,
                self.window_spec,
                self.val_idx,
            )

        # [MOD-6] 单独 validate 也构建 val_dataset
        if stage in ("validate",):
            self.val_dataset = DreamerDataset(
                self.io,
                self.global_manifest,
                self.window_spec,
                self.val_idx,
            )

        if stage in (None, "test"):
            self.test_dataset = DreamerDataset(
                self.io,
                self.global_manifest,
                self.window_spec,
                self.test_idx,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            # [MOD-7] DataLoader 稳定性/性能参数
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(self.num_workers > 0),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(self.num_workers > 0),
        )