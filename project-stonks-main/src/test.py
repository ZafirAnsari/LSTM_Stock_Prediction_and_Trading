import torch
import model as model
import trainer
import dataReader
from trader import Trader
import pandas as pd
import numpy as np
#Initialize models, set parameters
WINDOW_SIZE = 20
PREDICTION_OFFSET = 1
DATE_RANGE = 365 * 3
START_DATE = '2015-01-01'
HIDDEN_SIZE = 100
HIDDEN_COUNT = 1
STEP_COUNT = 1000
STOCK_FILE = 'YFinanceStockData/AAPL.csv'

input_data, label_data = dataReader.readStockData(STOCK_FILE,WINDOW_SIZE,PREDICTION_OFFSET,START_DATE,DATE_RANGE,binary=False,percent=True,normalize=True)

dates = pd.read_csv(STOCK_FILE,usecols=['date'])
dates = dates[START_DATE<dates['date']].to_numpy()
#Init each separate model
lstm = model.LSTMModel(torch.nn.ReLU(),WINDOW_SIZE,HIDDEN_SIZE,HIDDEN_COUNT)
lstm.init_weights()

#cnn = model.CNNModel(1,WINDOW_SIZE,kernel_size=5)
#cnn.init_weights()

##hybrid = model.HybridModel(WINDOW_SIZE,HIDDEN_SIZE,HIDDEN_COUNT,16,8,4)
#hybrid.init_weights()

#Train Models independently
optimizer = torch.optim.Adam
loss_function = torch.nn.MSELoss()
kwargs = {
    "lr" : 0.001,
}
LSTMTrainer = trainer.LSTMTrainer(optimizer,lstm,loss_function,**kwargs)
LSTMLosses = LSTMTrainer.train(input_data,label_data,STEP_COUNT,train=True)

#CNNTrainer = trainer.CNNTrainer(optimizer,cnn,loss_function,**kwargs)
#CNNLosses = CNNTrainer.train(input_data,label_data,STEP_COUNT,WINDOW_SIZE,6,train=True)

#HybridTrainer = trainer.HybridTrainer(optimizer,hybrid,loss_function,**kwargs)
#HybridLosses = HybridTrainer.train(input_data,label_data,STEP_COUNT,train=True)

test_data, test_labels = dataReader.readStockData(STOCK_FILE,WINDOW_SIZE,PREDICTION_OFFSET,'2018-01-01',DATE_RANGE,binary=False,normalize=True,percent=True)
unnorm_data, unnorm_labels = dataReader.readStockData(STOCK_FILE,WINDOW_SIZE,PREDICTION_OFFSET,'2018-01-01',DATE_RANGE,percent=True,normalize=False)
ELEMENT_COUNT = test_data.shape[0] - WINDOW_SIZE - PREDICTION_OFFSET + 1
LSTMShort = Trader(True)
LSTMLong = Trader(False)
CNNShort = Trader(True)
CNNLong = Trader(False)
HybridShort= Trader(True)
HybridLong = Trader(False)
StockMarket = [(LSTMShort,"Short LSTM"),(LSTMLong,"Long LSTM"),(CNNShort,"Short CNN"),(CNNLong,"Long CNN"),(HybridShort, "Short Hybrid"),(HybridLong,"Long Hybrid")]
for trader in StockMarket:
    trader[0].initialCost = unnorm_data[0][0][0]
lastPrice = 0
for i in range(ELEMENT_COUNT):
    #CnnPrediction = CNNTrainer.model(test_data[i],WINDOW_SIZE,6).item()
    LSTMprediction = LSTMTrainer.model(test_data[i]).item()
    #HybridPrediction = HybridTrainer.model(test_data[i]).item()
    day_of = unnorm_data[i][0][WINDOW_SIZE - 1].item()
    #next_price = CnnPrediction * day_of / 100
    #CNNShort.actOnPrediction(day_of,next_price,verbose=False)
    #CNNLong.actOnPrediction(day_of,next_price,verbose=False)
    next_price = LSTMprediction * day_of / 100
    LSTMShort.actOnPrediction(day_of,next_price,verbose=False,date=dates[i][0])
    LSTMLong.actOnPrediction(day_of,next_price,verbose=False,date=dates[i][0])
    #next_price = HybridPrediction * day_of / 100
    #HybridShort.actOnPrediction(day_of,next_price,verbose=False)
    #HybridLong.actOnPrediction(day_of,next_price,verbose=False)
    lastPrice = day_of


for trader in StockMarket:
    trader[0].closePositions(lastPrice,date=dates[ELEMENT_COUNT-1][0])
 #   print(trader[1])
 #   print("----------STATS---------")
 #   trader[0].printStats()
 #   print("------------------------")
#print(LSTMShort.transactions[0]['Close'])
LSTMShort.plotTrades(stockFile=STOCK_FILE)