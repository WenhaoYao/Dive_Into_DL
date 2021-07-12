import torch
from torch import nn
from d2l import torch as d2l
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' 

if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    net = nn.Sequential(nn.Flatten(), nn.Linear(784,10))
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight,std=0.01)
    net.apply(init_weights)
    loss = nn.CrossEntropyLoss()
    updater = torch.optim.SGD(net.parameters(), lr=0.1)
    d2l.train_ch3(net, train_iter, test_iter, loss,5 , updater)