import torch
import numpy
from ensemble import Ensemble
from ensembleTrainer import EnsembleTrainer
import dataReader
from trader import Trader
WINDOW_SIZE = 20
HIDDEN_SIZE = 128
LAYER_COUNT = 2
CNN_KERNEL = 6
HYBRID_PROJ = 16
HYBRID_CNN = 8
MLP_DEPTH = 4
STOCK_FILE = 'YFinanceStockData/QQQ.csv'
PREDICTION_OFFSET = 1
START_DATE = '2017-1-1'
DATE_RANGE = 365
STEPS = 1000


input_data, label_data = dataReader.readStockData(STOCK_FILE,WINDOW_SIZE,PREDICTION_OFFSET,START_DATE,DATE_RANGE,binary=False,percent=True,normalize=True)

optimizer = torch.optim.SGD
loss_func = torch.nn.MSELoss()

kwargs = {
    'lr' : 0.00001,
    'momentum': 0.8
}

ensembleModel = Ensemble(WINDOW_SIZE,HIDDEN_SIZE,LAYER_COUNT,CNN_KERNEL,HYBRID_PROJ,HYBRID_CNN,MLP_DEPTH)
ensembleTrainer = EnsembleTrainer(optimizer=optimizer,model=ensembleModel,loss_func=loss_func,**kwargs)
ensembleLosses = ensembleTrainer.train(input_data,label_data,STEPS)

shortTrader = Trader(True)
longTrader = Trader(False)

test_data, test_labels = dataReader.readStockData(STOCK_FILE,WINDOW_SIZE,PREDICTION_OFFSET,'2018-1-1',DATE_RANGE,binary=False,normalize=True,percent=True)
unnorm_data, unnorm_labels = dataReader.readStockData(STOCK_FILE,WINDOW_SIZE,PREDICTION_OFFSET,'2018-1-1',DATE_RANGE,percent=True,normalize=False)
ELEMENT_COUNT = test_data.shape[0] - WINDOW_SIZE - PREDICTION_OFFSET + 1
shortTrader.initialCost = unnorm_data[0][0][0]
longTrader.initialCost = unnorm_data[0][0][0]

lastPrice = 0
for i in range(ELEMENT_COUNT):
    Prediction = ensembleTrainer.model(test_data[i]).item()
    day_of = unnorm_data[i][0][WINDOW_SIZE - 1].item()
    next_price = Prediction * day_of / 100
    shortTrader.actOnPrediction(day_of,next_price,verbose=False)
    longTrader.actOnPrediction(day_of,next_price,verbose=False)
    lastPrice = day_of

shortTrader.closePositions(lastPrice)
longTrader.closePositions(lastPrice)

print("Long:")
longTrader.printStats()
print("Short:")
shortTrader.printStats()
