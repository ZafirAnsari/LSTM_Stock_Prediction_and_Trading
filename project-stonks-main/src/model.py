import torch

class LSTMModel(torch.nn.Module):
    
    def __init__(self,activation,input_size,hidden_size,layer_count):
        super().__init__()
        layers = []
        layers.append(torch.nn.LSTM(input_size,hidden_size,layer_count,proj_size=1))
        layers.append(activation)
        self.net = torch.nn.Sequential(*layers)

    def forward(self,input: torch.Tensor):
        x = self.net[0](input)
        #print(x[0][0])
        x = x[0][0]
        return self.net[1](x)
    
    def init_weights(self):
        for module in self.net.modules():
            if isinstance(module,torch.nn.LSTM):
                for item in module._parameters:
                    torch.nn.init.uniform_(module._parameters.get(item))

class CNNModel(torch.nn.Module):

    def __init__(self, in_channels,out_channels,kernel_size,**kwargs):
        super().__init__()
        layers = []
        layers.append(torch.nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=1))
        layers.append(torch.nn.Conv1d(in_channels=out_channels,out_channels=1,kernel_size=kernel_size,stride=1))
        self.net = torch.nn.Sequential(*layers)
    
    def forward(self,input: torch.Tensor,out_channels,kernel_size):
        maxPool = torch.nn.MaxPool1d(kernel_size=2)
        ReLU = torch.nn.ReLU()
        x = self.net[0](input)
        x = maxPool(ReLU(x))
        x = self.net[1](x)
        x = maxPool(ReLU(x))
        x = torch.mean(x)
        return x
    
    def init_weights(self):
        for module in self.net.modules():
            if isinstance(module,torch.nn.Conv1d):
                for item in module._parameters:
                    torch.nn.init.uniform_(module._parameters.get(item))


class HybridModel(torch.nn.Module):

    def __init__(self,input_size,hidden_size,layer_count,LSTMProj,ConvProj,ConvKernel):
        super().__init__()
        layers = []
        layers.append(torch.nn.LSTM(input_size,hidden_size,layer_count,proj_size=LSTMProj))
        layers.append(torch.nn.ReLU())#as given by paper
        layers.append(torch.nn.Conv1d(in_channels=1,out_channels=ConvProj,kernel_size=ConvKernel))
        layers.append(torch.nn.ReLU())
        self.net = torch.nn.Sequential(*layers)

    def forward(self,input: torch.Tensor):
        x = self.net[0](input)
        x = self.net[1](x[0])
        x = self.net[2](x)
        x = self.net[3](x)
        x = torch.mean(x)
        return x

    def init_weights(self):
        for module in self.net.modules():
            if isinstance(module,torch.nn.LSTM) or isinstance(module,torch.nn.Conv1d):
                for item in module._parameters:
                    torch.nn.init.uniform_(module._parameters.get(item))

    