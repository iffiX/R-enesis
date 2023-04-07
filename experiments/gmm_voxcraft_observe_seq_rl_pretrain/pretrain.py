import torch as t
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.utils.data import DataLoader, IterableDataset
from typing import Dict, Any
from renesis.env.virtual_shape import VirtualShapeGMMObserveSeqEnvironment
from experiments.gmm_voxcraft_observe_seq_rl_pretrain.model import Actor


class PretrainDataset(IterableDataset):
    def __init__(self, env, steps, sample_num, seed: int = 42):
        self.env = env
        self.steps = steps
        self.sample_num = sample_num
        self.seed = seed

    def __len__(self):
        return self.steps * self.sample_num

    def __getitem__(self, item):
        return


class Pretrainer(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.env = VirtualShapeGMMObserveSeqEnvironment(config["env_config"])
        self.model = Actor(
            self.env.observation_space,
            self.env.action_space,
            config["model"],
            **config["model"]["custom_model_config"],
        )

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.config["lr"])
