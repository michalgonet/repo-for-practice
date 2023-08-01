import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


class CTDataset(Dataset):
    def __init__(self, filepath):
        self.x, self.y = torch.load(filepath)
        self.x = self.x / 255
        self.y = F.one_hot(self.y, num_classes=10).to(float)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]


train_ds = CTDataset('Data\\MNIST\\processed\\training.pt')
test_ds = CTDataset('Data\\MNIST\\processed\\test.pt')

train_dl = DataLoader(train_ds, batch_size=5)
loss_fun = nn.CrossEntropyLoss()


# Network
class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.M1 = nn.Linear(28 ** 2, 100)
        self.M2 = nn.Linear(100, 50)
        self.M3 = nn.Linear(50, 10)
        self.R = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 ** 2)
        x = self.R(self.M1(x))
        x = self.R(self.M2(x))
        x = self.M3(x)
        return x.squeeze()


ff = MyNeuralNet()


def train_model(dl, f, n_epochs=2):
    opt = SGD(f.parameters(), lr=0.01)
    L = nn.CrossEntropyLoss()

    # Train model
    losses, epochs = [], []
    for epoch in range(n_epochs):
        print(f'Epoch {epoch}')
        N = len(dl)
        for i, (x, y) in enumerate(dl):
            # Update the weights of the network
            opt.zero_grad()
            loss_value = L(f(x), y)
            loss_value.backward()
            opt.step()
            # Store training data
            epochs.append(epoch + 1 / N)
            losses.append(loss_value.item())
    return np.array(epochs), np.array(losses)


epoch_data, loss_data = train_model(train_dl, ff)

epoch_data_avg = epoch_data.reshape(20, -1).mean(axis=1)
loss_data_avg = loss_data.reshape(20, -1).mean(axis=1)

xs, ys = test_ds[0:2000]
yhats = ff(xs).argmax(axis=1)

fig, ax = plt.subplots(10, 4, figsize=(10, 15))
for i in range(40):
    plt.subplot(10, 4, i + 1)
    plt.imshow(xs[i])
    plt.title(f'Prediction = {yhats[i]}')
fig.tight_layout()
plt.show()
