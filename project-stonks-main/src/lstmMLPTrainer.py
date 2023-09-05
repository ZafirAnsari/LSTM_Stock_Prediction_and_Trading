from tqdm import tqdm
import time
import numpy as np
import torch
class lstmMLPTrainer:

    def __init__(self, optimizer, model, loss_func, **kwargs):

        self.optimizer = optimizer(model.parameters(), **kwargs)
        self.model = model
        self.loss_func = loss_func
        self.epoch = 0
        self.start_time = None

    def run_one_epoch(self, x, y, train=True):
        if train:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()
        # print(x.shape)
        for i in x:
            predictions = self.model(torch.roll(x, 1, 1))
            # print(predictions)
            loss = self.loss_func(predictions, torch.tensor([y[0]]))
            # print(loss)
            if train:
                loss.backward()
                self.optimizer.step()
            return loss.item()

    def train(self, x, y, steps, train=True):
        self.start_time = time.time()
        losses = []
        for step in tqdm(range(steps)):
            for j in range(x.size()[0]):
                # print(x.size())
                loss = self.run_one_epoch(x[j], y[j], train=train)
                losses.append(loss)
                if train:
                    self.epoch += 1
        return losses[-1]