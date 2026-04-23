import torch
import torch.nn as nn
import pytorch_lightning as pl


class TSClassifier(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lr = cfg.train.learning_rate

        # 最小可跑模型：先别上RNN/Transformer
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 100, 2),  # 对应你当前假数据形状
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.long())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)