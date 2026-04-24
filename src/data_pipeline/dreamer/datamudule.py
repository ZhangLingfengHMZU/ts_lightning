# src/data_pipeline/dreamer/datamudule.py
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from typing import Optional

from src.data_pipeline.dreamer.tools.build_manifest import build_global_manifest
from src.data_pipeline.dreamer.mat_io import DreamerMatIO, WindowSpec

class DreamerDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # 只取第一小步需要的3个关键字段
        self.dataset_name = cfg.dataset.dataset_name
        self.batch_size = cfg.train.batch_size
        self.num_workers = cfg.train.num_workers

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

            self.dream_io = DreamerMatIO(
                mat_path=self.cfg.dataset.mat_path,
                verify_compressed_data_integrity=False,
            )
            self.window_spec = WindowSpec(
                win_sec=self.cfg.dataset.win_sec,
                stride_sec=self.cfg.dataset.stride_sec,
            )

            self.global_manifest = build_global_manifest(
                io=self.dream_io,
                window_spec=self.window_spec,
                is_force_rebuild=self.cfg.dataset.is_force_rebuild,
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        return None

    def setup(self, stage: Optional[str] = None) -> None:
        return None
    def train_dataloader(self) -> DataLoader:
        return 
    def val_dataloader(self) -> DataLoader:
        return 
