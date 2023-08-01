import sys
import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from urllib.parse import urlparse
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


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


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


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

alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

with mlflow.start_run():
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_ds[0], train_ds[1])

    predicted_qualities = lr.predict(test_ds[0])

    (rmse, mae, r2) = eval_metrics(test_ds[1], predicted_qualities)

    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    predictions = lr.predict(train_ds[0])
    signature = infer_signature(train_ds[0], predictions)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":
        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(
            lr, "model", registered_model_name="ElasticnetWineModel", signature=signature
        )
    else:
        mlflow.sklearn.log_model(lr, "model", signature=signature)
