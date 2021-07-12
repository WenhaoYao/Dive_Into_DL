import random
import torch

def sythetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))



def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(num_examples, i + batch_size)])
        yield features[batch_indices], labels[batch_indices]



def linreg(X, w, b):
    return torch.matmul(X, w) + b

def loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2 / 2

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= param.grad * lr / batch_size
            param.grad.zero_()



true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = sythetic_data(true_w, true_b, 1000)
w = torch.normal(0, 0.01, size = (2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
batch_size = 10

lr = 0.01
num_epochs = 5
net = linreg

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_l = loss(net(features, w, b),labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')