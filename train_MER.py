import hydra
from omegaconf import OmegaConf

import torch
import pytorch_lightning as pl

from src.data_pipeline.dreamer.datamudule import DreamerDataModule



@hydra.main(
    version_base=None,
    config_path="configs/MER",
    config_name="config",
)
def main(cfg):
    print("=== cfg ===")
    print(cfg)
    dm = DreamerDataModule(cfg)
    # 1) 准备数据（生成或复用 manifest）
    dm.prepare_data()
    # 2) setup fit 阶段（切分 train/val 并实例化 dataset）
    dm.setup(stage="fit")
    # 3) 拿 dataloader
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    print(f"train_dataset size = {len(dm.train_dataset)}")
    print(f"val_dataset size   = {len(dm.val_dataset)}")
    print(f"train_batches      = {len(train_loader)}")
    print(f"val_batches        = {len(val_loader)}")
    # 4) 抽一个 batch 看 shape / dtype（无模型也能验证）
    x_train, y_train = next(iter(train_loader))
    x_val, y_val = next(iter(val_loader))
    print("\n=== train batch ===")
    print(f"x_train shape: {tuple(x_train.shape)}, dtype: {x_train.dtype}")
    print(f"y_train shape: {tuple(y_train.shape)}, dtype: {y_train.dtype}")
    print(f"y_train unique: {torch.unique(y_train)}")
    print("\n=== val batch ===")
    print(f"x_val shape: {tuple(x_val.shape)}, dtype: {x_val.dtype}")
    print(f"y_val shape: {tuple(y_val.shape)}, dtype: {y_val.dtype}")
    print(f"y_val unique: {torch.unique(y_val)}")
    print("\n[OK] DataModule smoke test passed.")
if __name__ == "__main__":
    main()