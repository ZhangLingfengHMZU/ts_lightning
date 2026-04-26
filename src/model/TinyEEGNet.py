import torch
import torch.nn as nn
import pytorch_lightning as pl

class TinyEEGNet(pl.LightningModule):
    def __init__(self, in_ch=14, n_classes=2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.feat = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, n_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):          # x: [B, C, T]
        z = self.feat(x).squeeze(-1)
        return self.head(z)

    def _step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.long())
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, _): return self._step(batch, "train")
    def validation_step(self, batch, _): self._step(batch, "val")
    def test_step(self, batch, _): self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)