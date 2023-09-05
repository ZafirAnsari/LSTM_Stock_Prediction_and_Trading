from tqdm import tqdm
import time
import numpy as np

class EnsembleTrainer:

    def __init__(self,optimizer,model,loss_func,**kwargs):

        self.optimizer = optimizer(model.parameters(),**kwargs)
        self.model = model
        self.loss_func = loss_func
        self.epoch = 0
        self.start_time = None

    def run_one_epoch(self,x,y,train=True,techInd=[]):
        if train:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()
        predictions = self.model(x,techInd=techInd)
        loss = self.loss_func(predictions,y)
        if train:
            loss.backward()
            self.optimizer.step()
        return loss.item()
    
    def train(self,x,y,steps,train=True,techInd = []):
        self.start_time = time.time()
        losses = []
        for step in range(steps):
            for j in range(x.size()[0]):
                loss = self.run_one_epoch(x[j],y[j],train = train,techInd=techInd)
                losses.append(loss)
                if train:
                    self.epoch += 1
        return losses[-1]