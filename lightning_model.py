import torch
import torch.nn as nn
import lightning as L
from torch import optim

import numpy as np

from networks import networks

class LitHSGCNN(L.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params

        print("Parameters:")
        for k, v in params.items():
            print(f"{k}: {v}")

        self._set_seed()
        self._get_net()
        self._get_loss()

    def _set_seed(self):
        if self.params["manual_seed"] != -1:
            torch.manual_seed(self.params["manual_seed"])
            np.random.seed(self.params["manual_seed"])
    
    def _get_net(self):
        """
        Loads network based on parameter network name.
        """
        network_name = self.params["network_name"]
        num_classes = self.params["num_classes"]
        n_groups_hue = self.params["n_groups_hue"]
        n_groups_saturation = self.params["n_groups_saturation"]

        network_class = networks.parse_name(network_name)
        self.net = network_class(num_classes=num_classes,
                                    n_groups_hue=n_groups_hue,
                                    n_groups_saturation=n_groups_saturation,
                                    ours=self.params["ours"])
        
        # log number of parameters in model
        # n_params = sum(p.numel() for p in self.net.parameters())
    
    def _get_loss(self):
        loss_name = self.params["loss_name"]
        if loss_name == "cross_entropy":
            self.loss = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Loss name {loss_name} not recognized.")

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch[0], batch[1]   # nb batch may have more than two elements
        y_hat = self.net(x)
        loss = self.loss(y_hat, y)
        # Logging to TensorBoard
        self.log("train_loss", loss)

        # Calculate accuracy
        _, predicted = y_hat.max(1)
        correct = predicted.eq(y).sum().item()
        self.log("train_accuracy", correct / y.size(0))

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.net(x)
        loss = self.loss(y_hat, y)
        # Logging to TensorBoard
        self.log("val_loss", loss)

        # Calculate accuracy
        _, predicted = y_hat.max(1)
        correct = predicted.eq(y).sum().item()
        self.log("val_accuracy", correct / y.size(0))

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch[0], batch[1]
        y_hat = self.net(x)
        loss = self.loss(y_hat, y)
        # Logging to TensorBoard
        self.log("test_loss", loss)

        # Calculate accuracy
        _, predicted = y_hat.max(1)
        correct = predicted.eq(y).sum().item()
        self.log("test_accuracy", correct / y.size(0))

    def configure_optimizers(self):
        lr = self.params["lr"]

        if self.params["optimizer"] == "adam":
            optimizer = optim.Adam(self.parameters(), lr=lr)
        elif self.params["optimizer"] == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Optimizer {self.params['optimizer']} not recognized.")
        
        if self.params["scheduler"] == "none":
            return optimizer
        elif self.params["scheduler"] == "cosine":
            if "n_epochs" in self.params:
                T_max = self.params["n_epochs"]
            else:
                T_max = 10
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            raise ValueError(f"Scheduler {self.params['scheduler']} not recognized.")