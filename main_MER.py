import hydra
from omegaconf import OmegaConf

import pytorch_lightning as pl

from src.data_pipeline.dreamer.datamudule import DreamerDataModule



@hydra.main(
    version_base=None,
    config_path="configs/MER", 
    config_name="config"
)
def main(cfg):
    print(cfg)
    datamodule = DreamerDataModule(cfg)
    datamodule.prepare_data()
if __name__ == "__main__":
    main()