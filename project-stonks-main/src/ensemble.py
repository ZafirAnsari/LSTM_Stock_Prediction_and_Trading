import torch

class Ensemble(torch.nn.Module):

    def __init__(self,window_size,lstm_hidden_size,lstm_layer_count,cnn_kernel_size,hybrid_projection_lstm,hybrid_projection_cnn,mlp_depth,techData=[]):
        super().__init__()
        #LSTM Only Layer
        self.mlpdepth = mlp_depth
        layers = []
        layers.append(torch.nn.LSTM(window_size,lstm_hidden_size,lstm_layer_count,proj_size=1))
        layers.append(torch.nn.ReLU())
        #CNN Only Layer
        layers.append(torch.nn.Conv1d(in_channels=1,out_channels=16,kernel_size=cnn_kernel_size,stride=1))
        layers.append(torch.nn.Conv1d(in_channels=16,out_channels=1,kernel_size=cnn_kernel_size,stride=1))
        #CNN LSTM Hybrid Layer
        layers.append(torch.nn.LSTM(window_size,lstm_hidden_size,lstm_layer_count,proj_size=hybrid_projection_lstm))
        layers.append(torch.nn.ReLU())#as given by paper
        layers.append(torch.nn.Conv1d(in_channels=1,out_channels=hybrid_projection_cnn,kernel_size=cnn_kernel_size))
        layers.append(torch.nn.ReLU())
        #MLP Network
        assert mlp_depth > 1, "MLP depth must be > 1"
        input_dims = 3 + len(techData)
        middle_dims = 10
        layers.append(torch.nn.Linear(input_dims,middle_dims))
        for _ in range(mlp_depth - 2):
            layers.append(torch.nn.Linear(middle_dims,middle_dims))
        #Final MLP Condensation layer + Final Activation
        layers.append(torch.nn.Linear(middle_dims,1))
        layers.append(torch.nn.ReLU())
        self.net = torch.nn.Sequential(*layers)

    def init_weights(self):
        for module in self.net.modules():
            if isinstance(module,torch.nn.LSTM) or isinstance(module,torch.nn.Conv1d) or isinstance(module,torch.nn.Linear):
                for item in module._parameters:
                    torch.nn.init.uniform_(module._parameters.get(item))

    def forward(self,input,techInd=[]):
        #Evaluate parallel modules
        #LSTM First
        lstmOut = self.net[0](input)
        lstmOut = lstmOut[0][0]
        lstmOut = self.net[1](lstmOut)
        #CNN
        maxPool = torch.nn.MaxPool1d(kernel_size=2)
        ReLU = torch.nn.ReLU()
        cnnOut = self.net[2](input)
        cnnOut = maxPool(ReLU(cnnOut))
        cnnOut = self.net[3](cnnOut)
        cnnOut = maxPool(ReLU(cnnOut))
        cnnOut = torch.mean(cnnOut)
        #Hybrid
        hybridOut = self.net[4](input)
        hybridOut = self.net[5](hybridOut[0])
        hybridOut = self.net[6](hybridOut)
        hybridOut = self.net[7](hybridOut)
        hybridOut = torch.mean(hybridOut)
        MLPIn = torch.tensor([hybridOut,cnnOut,lstmOut])
        if techInd:
            MLPIn.append(techInd)
        MLPOut = self.net[8](MLPIn)
        currInd = 8
        if self.mlpdepth > 2:
            for i in range(self.mlpdepth - 2):
                MLPOut = self.net[i + 9](MLPOut)
                currInd = i + 9
        #Final MLP layer is result
        return self.net[currInd + 1](MLPOut)
        
        
