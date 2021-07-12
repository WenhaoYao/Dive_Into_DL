import torch
from torch.nn import init
from torch.nn.modules.linear import Linear
from torch.utils import data
from torch import nn

def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def load_array(data_arrays, batch_size, is_train = True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle = is_train)

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

data_iter = load_array((features, labels), 10)
net = nn.Sequential(nn.Linear(2, 1))
def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weight)
loss = nn.MSELoss()

trainler = torch.optim.SGD(net.parameters(), lr=0.03)
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainler.zero_grad()
        l.backward()
        trainler.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')


     