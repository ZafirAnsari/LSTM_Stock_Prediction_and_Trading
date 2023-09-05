import torch
import model
import trainer
import dataReader
from trader import Trader

#from pytorch_lightning.callbacks.early_stopping import EarlyStopping
#Hyperparams
WINDOW_SIZE = 20
PREDICTION_OFFSET = 1
DATE_RANGE = 1000
START_DATE = '2017-12-6'
HIDDEN_SIZE = 100
HIDDEN_COUNT = 1
STEP_COUNT = 1000
STOCK_FILE = 'YFinanceStockData/BTC.csv'

lstm = model.LSTMModel(torch.nn.ReLU(),WINDOW_SIZE,HIDDEN_SIZE,HIDDEN_COUNT)
lstm.init_weights()

cnn = model.CNNModel(1,WINDOW_SIZE,kernel_size=5)
cnn.init_weights()

hybrid = model.HybridModel(WINDOW_SIZE,HIDDEN_SIZE,HIDDEN_COUNT)
hybrid.init_weights()


input_data, label_data = dataReader.readStockData(STOCK_FILE,WINDOW_SIZE,PREDICTION_OFFSET,START_DATE,DATE_RANGE,binary=False,percent=True,normalize=True)
loss_function = torch.nn.MSELoss()
input2 = input_data[0]
print(hybrid(input_data[0]))

optimizer = torch.optim.SGD

kwargs = {
    "lr" : 0.00001,
    "momentum" : 0.8
}

testTrainer = trainer.LSTMTrainer(optimizer,lstm,loss_function,**kwargs)
CNNTrainer = trainer.CNNTrainer(optimizer,cnn,loss_function,**kwargs)
CNNlosses = CNNTrainer.train(input_data,label_data,STEP_COUNT,WINDOW_SIZE,6,train=True)
losses = testTrainer.train(input_data,label_data,STEP_COUNT,train=True)

test_data, test_labels = dataReader.readStockData(STOCK_FILE,WINDOW_SIZE,PREDICTION_OFFSET,'2019-01-01',DATE_RANGE,binary=False,normalize=True,percent=True)
unnorm_data, unnorm_labels = dataReader.readStockData(STOCK_FILE,WINDOW_SIZE,PREDICTION_OFFSET,'2019-01-01',DATE_RANGE,percent=True,normalize=False)
ELEMENT_COUNT = test_data.shape[0] - WINDOW_SIZE - PREDICTION_OFFSET + 1
WarrenBuffett = Trader(False)
WallStreetBets = Trader(True)
WarrenBuffett.initialCost = unnorm_data[0][0][0]
WallStreetBets.initialCost = unnorm_data[0][0][0]
lastPrice = 0
for i in range(ELEMENT_COUNT):
    CnnPrediction = CNNTrainer.model(test_data[i],WINDOW_SIZE,6).item()
    prediction = testTrainer.model(test_data[i]).item()
    day_of = unnorm_data[i][0][WINDOW_SIZE - 1].item()
    next_price = CnnPrediction * day_of / 100
    WarrenBuffett.actOnPrediction(day_of,next_price,verbose=False)
    WallStreetBets.actOnPrediction(day_of,next_price,verbose=False)
    lastPrice = day_of


WarrenBuffett.closePositions(lastPrice)
WallStreetBets.closePositions(lastPrice)
print("-----")
WarrenBuffett.printStats()
print("-----")
WallStreetBets.printStats()