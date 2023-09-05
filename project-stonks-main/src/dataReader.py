
import torch
import numpy as np
import pandas as pd
import datetime
#Takes stock data -> (X,Y) where x: (input_window x Element Count) and y: (1 x element count)
def readStockData(filename,input_window,prediction_offset,firstdate,date_range,binary=False,percent=False,normalize=True,TA=[]):
    """
    filename: csv of individual stock
    input_window: number of data points per prediction (window size)
    prediction_offset: data points from last input being predicted
    firstdate: start of data pull
    date_range: number of datapoints for this dataset
    """
    if len(TA) == 0:
        data = pd.read_csv(filename)
        data['date'] = pd.to_datetime(data['date'])
        start = pd.to_datetime(firstdate)
        end = start + datetime.timedelta(days=date_range)
        data = data[start<data['date']]
        data = data[end>data['date']]
        element_count = data.shape[0] - input_window - prediction_offset + 1
        data_all = data['close'].to_numpy()
        stdev = np.std(data_all)
        avg = np.mean(data_all)
        x = np.empty((element_count,1,input_window))
        y = np.empty((element_count,1 if binary else 1))
        for i in range(0,element_count):
            tempTensor = torch.zeros(1,input_window)
            tempTensor[0] = torch.tensor(data['close'][i:i+input_window].to_numpy())
            x[i] = tempTensor
            torch.tensor(x[i])
            yind = i+input_window+prediction_offset
            if binary:
                pred_val = data['close'][yind-1:yind].to_numpy()
                cur_val = x[i][0][input_window-1]
                diff = pred_val - cur_val
                y[i] = [0 if diff < 0 else 1]#,1 if diff < 0 else 0] #Logits for BCE
            elif percent:
                pred_val = data['close'][yind-1:yind].to_numpy()
                cur_val = x[i][0][input_window-1]
                y[i] = pred_val*100/cur_val
            else:
                y[i] = data['close'][yind-1:yind].to_numpy()
        if normalize:
            for j in x:
                for k in range(input_window):
                     j[0][k] = (j[0][k] - avg)/stdev

        x = torch.tensor(x,dtype=torch.float32)
        # print(x, "reader")
        if binary:
            y = torch.tensor(y,dtype=torch.float32)
        else:
            y = torch.tensor(y,dtype=torch.float32)
        return (x,y)
    else:
        data = pd.read_csv(filename)
        data['date'] = pd.to_datetime(data['date'])
        start = pd.to_datetime(firstdate)
        end = start + datetime.timedelta(days=date_range)
        data = data[start < data['date']]
        data = data[end > data['date']]
        element_count = data.shape[0] - input_window - prediction_offset + 1
        data_stds = data.std()
        data_avgs = data.mean()
        x = np.empty((element_count, 1+len(TA), input_window))
        y = np.empty((element_count, 1 if binary else 1))
        for i in range(0, element_count):
            tempTensor = torch.zeros(1+len(TA), input_window)
            for ind, val in enumerate(["close"] + TA):
                tempTensor[ind] = torch.tensor(data[val][i:i + input_window].to_numpy())
            x[i] = tempTensor
            torch.tensor(x[i])
            yind = i + input_window + prediction_offset
            if binary:
                pred_val = data['close'][yind - 1:yind].to_numpy()
                cur_val = x[i][0][input_window - 1]
                diff = pred_val - cur_val
                y[i] = [0 if diff < 0 else 1]  # ,1 if diff < 0 else 0] #Logits for BCE
            elif percent:
                pred_val = data['close'][yind - 1:yind].to_numpy()
                cur_val = x[i][0][input_window - 1]
                y[i] = pred_val * 100 / cur_val
            else:
                y[i] = data['close'][yind - 1:yind].to_numpy()
        if normalize:
            for j in x:
                for k in range(input_window):
                    for ind, l in enumerate(["close"] + TA):
                        j[ind][k] = (j[ind][k] - data_avgs[l]) / data_stds[l]

        x = torch.tensor(x, dtype=torch.float32)
        if binary:
            y = torch.tensor(y, dtype=torch.float32)
        else:
            y = torch.tensor(y, dtype=torch.float32)
        return (x, y)

#xval, yval = readStockData('StockDataDaily/AA.csv',10,7,'2017-1-1',180,binary=False,normalize=True,percent=True)
def calculateTechIndicators(x):
    pass