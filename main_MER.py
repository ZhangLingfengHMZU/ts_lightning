import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from src.data_pipeline.dreamer.datamodule import DreamerDataModule
from src.model.TinyEEGNet import TinyEEGNet


@hydra.main(
    version_base=None,
    config_path="configs/MER",
    config_name="config",
)
def main(cfg):
    print("=== MER Run Summary ===")
    print(OmegaConf.to_yaml(cfg))

    # 1) DataModule
    dm = DreamerDataModule(cfg)
    print(dm.summary())

    # 2) Model
    model = TinyEEGNet(
        in_ch=14,      # 你的 EEG 通道数
        n_classes=2,   # 二分类
        lr=1e-3,
    )

    # 3) Logger (TensorBoard)
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name="mer_tinyeegnet",
    )

    # 4) Trainer (先用最小可跑配置)
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=1,
        logger=logger,
        log_every_n_steps=10,
        deterministic=True,
        enable_checkpointing=False,  # 先跑通流程
    )

    # 5) Fit / Test
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()