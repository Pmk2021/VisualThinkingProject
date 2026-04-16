import torch
import wandb


class Trainer:
    def __init__(self, model, optimizer, dataset, device, args):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.device = device
        self.args = args

    def train_single_epoch(self):
        pass

    def validate(self):
        pass

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.train_single_epoch()
            # Log metrics to Weights & Biases
            wandb.log({"epoch": epoch})
