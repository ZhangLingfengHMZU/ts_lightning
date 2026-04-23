import hydra
from omegaconf import OmegaConf

import pytorch_lightning as pl

from src.ts_toy_classification.data_pipeline.datamodule import TSDataModule
from src.ts_toy_classification.models.toy_classification_module import TSClassifier

@hydra.main(
    version_base=None,
    config_path="configs/ts_toy_classification", 
    config_name="config"
)
def main(cfg):
    print("=== Run Summary ===")
    print(f"experiment_name: {cfg.experiment_name}")
    print(f"dataset: {cfg.dataset.dataset_name}")
    print(
        f"model: input_dim={cfg.model.input_dim}, hidden_dim={cfg.model.hidden_dim}, "
        f"num_layers={cfg.model.num_layers}, head_dim={cfg.model.head_dim}, dropout={cfg.model.dropout}"
    )
    print(
        f"train: batch_size={cfg.train.batch_size}, lr={cfg.train.learning_rate}, "
        f"weight_decay={cfg.train.weight_decay}, num_workers={cfg.train.num_workers}"
    )
    print(
        f"trainer: accelerator={cfg.trainer.accelerator}, devices={cfg.trainer.devices}, "
        f"precision={cfg.trainer.precision}, max_epochs={cfg.trainer.max_epochs}"
    )
    print(f"paths: output={cfg.paths.output_dir}")
    print(f"       logs={cfg.paths.log_dir}")
    print(f"       ckpt={cfg.paths.checkpoint_dir}")
    print("===================")

    datamodule = TSDataModule(cfg)
    print(datamodule.summary())
    model = TSClassifier(cfg)
    # print(model.summary())
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
    )
    trainer.fit(model, datamodule)
if __name__ == "__main__":
    main()