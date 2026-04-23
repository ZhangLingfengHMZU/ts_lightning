# src/ts_toy_classification/datamodule.py
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from typing import Optional

class TSDataModule(pl.LightningDataModule):
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
            f"TSDataModule(dataset_name={self.dataset_name}, "
            f"batch_size={self.batch_size}, "
            f"num_workers={self.num_workers})"
        )
    
    def prepare_data(self) -> None:
        if self.dataset_name == "sMNIST":
            print("假装在下载: Preparing sMNIST dataset...")

        elif self.dataset_name == "synthetic":
            print("假装在下载: Preparing synthetic dataset...")

        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        return None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            # Minimal dummy classification data for wiring checks.
            x = torch.randn(100, 3, 100)
            y = torch.randint(low=0, high=2, size=(100,))

            x_train, y_train = x[:80], y[:80]
            x_val, y_val = x[80:], y[80:]

            self.train_dataset = TensorDataset(x_train, y_train)
            self.val_dataset = TensorDataset(x_val, y_val)
        return None
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
