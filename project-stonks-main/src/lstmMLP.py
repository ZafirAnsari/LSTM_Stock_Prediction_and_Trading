import torch

class lstmMLP(torch.nn.Module):

    def __init__(self, lstmModel: torch.nn.Module, mlp_depth, hidden_dims=10, techData=[]):
        super().__init__()
        # LSTM Only Layer
        self.mlpdepth = mlp_depth
        self.LSTM = lstmModel
        layers = []
        # MLP Network
        assert mlp_depth > 1, "MLP depth must be > 1"
        input_dims = 1 + len(techData)
        layers.append(torch.nn.Linear(input_dims, hidden_dims))
        for _ in range(mlp_depth - 2):
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(hidden_dims, hidden_dims))
        # Final MLP Condensation layer + Final Activation
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_dims, 1))
        layers.append(torch.nn.ReLU())
        self.net = torch.nn.Sequential(*layers)

    def init_weights(self):
        for module in self.net.modules():
            if isinstance(module, torch.nn.LSTM) or isinstance(module, torch.nn.Linear):
                for item in module._parameters:
                    torch.nn.init.uniform_(module._parameters.get(item))

    def forward(self, LSTMinput):
        # Evaluate parallel modules
        # LSTM First
        # print(LSTMinput.shape)
        lstmOut = self.LSTM(LSTMinput[0])
        # LSTMinput = torch.tensor([lstmOut.item()] * LSTMinput.shape[1]), torch.tensor(LSTMinput[1:])

        MLPIn = torch.cat((torch.tensor([lstmOut.item()] * LSTMinput.shape[1]).unsqueeze(0), torch.tensor(LSTMinput[1:])), 0)
        # print(MLPIn)

        # print(MLPIn[:,0])
        # MLPIn = torch.cat((torch.tensor([lstmOut.item()]), torch.tensor(LSTMinput[:][0])))
        #
        # print(MLPIn.shape)

        MLPOut = self.net(MLPIn[:,0])
        return MLPOut


