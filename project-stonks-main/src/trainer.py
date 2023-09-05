import torch
import numpy as np
from dataReader import readStockData
import time
from tqdm import tqdm

class LSTMTrainer:

    def __init__(self,optimizer,model,loss_func,**kwargs):

        self.optimizer = optimizer(model.parameters(),**kwargs)
        self.model = model
        self.loss_func = loss_func
        self.epoch = 0
        self.start_time = None

    def run_one_epoch(self,x,y,train=True):
        if train:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()
        predictions = self.model(x)
        loss = self.loss_func(predictions,y)
        if train:
            loss.backward()
            self.optimizer.step()
        return loss.item()
    
    def train(self,x,y,steps,train=True):
        self.start_time = time.time()
        losses = []
        for step in tqdm(range(steps)):
            for j in range(x.size()[0]):
                loss = self.run_one_epoch(x[j],y[j],train = train)
                losses.append(loss)
                if train:
                    self.epoch += 1
        return losses[-1]

class CNNTrainer:

    def __init__(self,optimizer,model,loss_func,**kwargs):

        self.optimizer = optimizer(model.parameters(),**kwargs)
        self.model = model
        self.loss_func = loss_func
        self.epoch = 0
        self.start_time = None

    def run_one_epoch(self,x,y,window_size,kernel_size,train=True):
        if train:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()
        predictions = self.model(x,window_size,kernel_size)
        loss = self.loss_func(predictions,y[0])
        if train:
            loss.backward()
            self.optimizer.step()
        return loss.item()
    
    def train(self,x,y,steps,window_size,kernel_size,train=True):
        self.start_time = time.time()
        losses = []
        for step in tqdm(range(steps)):
            for j in range(x.size()[0]):
                loss = self.run_one_epoch(x[j],y[j],window_size,kernel_size,train = train)
                losses.append(loss)
            if train:
                self.epoch += 1
        return losses[-1]

class HybridTrainer:

    def __init__(self,optimizer,model,loss_func,**kwargs):

        self.optimizer = optimizer(model.parameters(),**kwargs)
        self.model = model
        self.loss_func = loss_func
        self.epoch = 0
        self.start_time = None

    def run_one_epoch(self,x,y,train=True):
        if train:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()
        predictions = self.model(x)
        loss = self.loss_func(predictions,torch.tensor([y[0]]))
        if train:
            loss.backward()
            self.optimizer.step()
        return loss.item()
    
    def train(self,x,y,steps,train=True):
        self.start_time = time.time()
        losses = []
        for step in tqdm(range(steps)):
            for j in range(x.size()[0]):
                loss = self.run_one_epoch(x[j],y[j],train = train)
                losses.append(loss)
                if train:
                    self.epoch += 1
        return losses[-1]