import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()

        # TODO: How to modify this model to achieve better performance?
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    ''' 
    Given input of size (batch_size x input_dim), compute output of the network 
    '''
    def forward(self, x):
        return self.net(x).squeeze(1)

    ''' 
    Calculate loss 
    '''
    def cal_loss(self, pred, target):
        # TODO: you may implement L1/L2 regularization here
        return self.criterion(pred, target)
