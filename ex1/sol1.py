import pandas as pd
import numpy as np
import torch.cuda
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils

device = "cuda" if torch.cuda.is_available() else "cpu"
neg = pd.read_csv('Data/neg_A0201.txt', names=['seq'])  # negative for covid samples
neg = neg[:100]
pos = pd.read_csv('Data/pos_A0201.txt', names=['seq'])  # positive for covid samples
pos = pos[:100]
amino_letters = 'ACDEFGHIKLMNPQRSTVWY'  # Amino acids signs
one_hot_values = [str(i) + acid for i in range(9) for acid in amino_letters]  # one hot features


def generate_X(peptid_df):
    """
    generates a sample matrix with all the relevant dummy variables from the given peptid dataframe
    :param peptid_df: a dataframe with the peptids
    :return: altered df
    """
    peptid_df[one_hot_values] = 0  # adding all one hot encoding labels
    for i, peptid_name in enumerate(peptid_df['seq']):
        peptid_one_hot = [str(i) + acid for i, acid in enumerate(peptid_name)]  # columns that should be assigned 1
        peptid_df.at[i, peptid_one_hot] = 1
    return peptid_df.drop(["seq"],axis=1)


def preprocess():
    """

    :return: training and test X,y
    """
    X_pos, X_neg = generate_X(pos), generate_X(neg)
    y_pos, y_neg = pd.DataFrame(np.ones(X_pos.shape[0])), pd.DataFrame(-np.ones(X_neg.shape[0]))
    X, y = pd.concat([X_pos, X_neg]), pd.concat([y_pos, y_neg])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    X_train_tensor, X_test_tensor = torch.tensor(X_train.values), torch.tensor(X_test.values)
    y_train_tensor, y_test_tensor = torch.tensor(y_train.values), torch.tensor(y_test.values)
    train = data_utils.TensorDataset(X_train_tensor, y_train_tensor)
    test = data_utils.TensorDataset(X_test_tensor, y_test_tensor)
    return train, test


class MLP(nn.Module):
    """
      Multilayer Perceptron.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(180, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        forward pass
        :param x: network input
        :return: network output
        """
        return self.layers(x)


def train(dataloader, nn_model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    nn_model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        prediction = model(X)
        loss = loss_fn(prediction, y)

        # backprop:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():  # use this line when not using backprop (pyTorch performs some optimizations)
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    epochs, batch_size, learning_rate = 5, 100, 1e-3
    train_data, test_data = preprocess()
    model = MLP().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_dataloader = DataLoader(train_data, batch_size)
    test_dataloader = DataLoader(test_data, batch_size)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("WO-HA we're done")
