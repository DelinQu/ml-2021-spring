import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_dim, lamb=0.00075):
        super(NeuralNet, self).__init__()
        self.lamb = lamb

        # TODO: How to modify this model to achieve better performance ?
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),         # BN, 加速模型训练
            nn.Dropout(p=0.2),          # Dropout, 减小过拟合, 注意不能在BN之前
            nn.LeakyReLU(),             # 更换激活函数
            nn.Linear(32,1)
        )
        # self.net = nn.Sequential(
        #     nn.Linear(input_dim, 64),
        #     nn.ReLU(),
        #     nn.Linear(64,1)
        # )

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
        # return self.criterion(pred, target)
        regular_loss = 0
        for param in self.net.parameters():
            regular_loss += torch.sum(param ** 2)
        return self.criterion(pred, target) + self.lamb*regular_loss