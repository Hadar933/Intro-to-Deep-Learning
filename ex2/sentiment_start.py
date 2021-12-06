import torch
from matplotlib import pyplot as plt
from torch.nn.functional import pad
import torch.nn as nn
import numpy as np
import loader as ld
import torch.utils.data as data_utils

# some parameters to play with
batch_size = 32
output_size = 2
hidden_size = 64  # to experiment with
test_interval = 50

run_recurrent = False  # else run Token-wise MLP
use_RNN = False  # otherwise GRU
atten_size = 5  # atten > 0 means using restricted self atten

reload_model = False
num_epochs = 10
learning_rate = 0.01

train_dataset, test_dataset, num_words, input_size = ld.get_data_set(batch_size, toy=True)


class MatMul(nn.Module):
    """
    Special matrix multipication layer (like torch.Linear but can operate on arbitrary sized
    tensors and considers its last two indices as the matrix.)
    """

    def __init__(self, in_channels, out_channels, use_bias=True):
        super(MatMul, self).__init__()
        self.matrix = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(in_channels, out_channels)),
                                         requires_grad=True)
        if use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(1, 1, out_channels), requires_grad=True)

        self.use_bias = use_bias

    def forward(self, x):
        x = torch.matmul(x, self.matrix)
        if self.use_bias:
            x = x + self.bias
        return x


class ExRNN(nn.Module):
    """
    this class is an implementation of a basic Elman network
    """

    def __init__(self, input_size, output_size, hidden_size):
        super(ExRNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.sigmoid = torch.sigmoid

        # RNN Cell weights
        self.W_in_hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_out = nn.Linear(hidden_size, output_size)

    def name(self):
        return "RNN"

    def forward(self, x, hidden_state):
        concat = torch.cat((x, hidden_state), 1)
        hidden = torch.tanh(self.W_in_hidden(concat))
        output = self.sigmoid(self.W_out(hidden))
        return output, hidden

    def init_hidden(self, bs):
        """
        :param bs: batch siz78e
        :return:
        """
        return torch.zeros(bs, self.hidden_size)


class ExGRU(nn.Module):
    """
    this class is an implementation of a GRU cell
    """

    def __init__(self, input_size, output_size, hidden_size):
        super(ExGRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.sigmoid = torch.sigmoid

        # GRU Cell weights
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)
        self.fully_connected = nn.Linear(hidden_size, output_size)

    def name(self):
        return "GRU"

    def forward(self, x, hidden_state):
        concat1 = torch.cat((hidden_state, x), 1)
        z = self.sigmoid(self.W_z.forward(concat1))
        r = self.sigmoid(self.W_r.forward(concat1))
        concat2 = torch.cat((r * hidden_state, x), 1)
        h_tilde = torch.tanh(self.W_h.forward(concat2))
        hidden = (1 - z) * hidden_state + z * h_tilde
        output = torch.sigmoid(self.fully_connected(hidden))
        return output, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size)


class ExMLP(nn.Module):
    """
    this class is an implementation of a basic MLP network
    """

    def __init__(self, input_size, output_size, hidden_size):
        super(ExMLP, self).__init__()

        self.ReLU = torch.nn.ReLU()

        # Token-wise MLP network weights
        self.layer1 = MatMul(input_size, hidden_size)
        self.layer2 = MatMul(hidden_size, output_size)

    def name(self):
        return "MLP"

    def forward(self, x):
        # Token-wise MLP network implementation

        x = self.layer1(x)
        x = self.ReLU(x)
        x = self.layer2(x)
        x = torch.sigmoid(x)
        return x


class ExLRestSelfAtten(nn.Module):
    """
    this class is an implementation of an MPL networh with Restricted Self Attention layer
    """

    def __init__(self, input_size, output_size, hidden_size):
        super(ExLRestSelfAtten, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.sqrt_hidden_size = np.sqrt(float(hidden_size))
        self.ReLU = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(2)

        # MLP layers
        self.layer1 = MatMul(input_size, hidden_size)
        self.layer2 = MatMul(hidden_size, output_size)

        # Attention learned weights
        self.W_q = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_k = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_v = MatMul(hidden_size, hidden_size, use_bias=False)

    def name(self):
        return "MLP_atten"

    def forward(self, x):
        x = self.layer1(x)
        x = self.ReLU(x)

        # generating x in offsets between -atten_size and atten_size 
        # with zero padding at the ends

        padded = pad(x, (0, 0, atten_size, atten_size, 0, 0))

        x_nei = []
        for k_T in range(-atten_size, atten_size + 1):
            x_nei.append(torch.roll(padded, k_T, 1))

        x_nei = torch.stack(x_nei, 2)
        x_nei = x_nei[:, atten_size:-atten_size, :]

        # x_nei has an additional axis that corresponds to the offset
        query = self.W_q(x)
        keys = self.W_k(x_nei)
        values = self.W_v(x_nei)
        keys_T = torch.transpose(keys, dim0=1, dim1=3)
        # query (32,100,64)
        #         a  b  c
        # keys_T (32,64,11,100)
        #         a  c  d   f
        # QK_T (32,100,100)
        #        a  b   f
        QK_T = torch.einsum('abc,acdf->abf', [query, keys_T])
        atten_weights = self.softmax(QK_T) / self.sqrt_hidden_size
        # atten_weights (32,100,100)
        #                 a  b   f
        # values      (32,100,11,64)
        #               a  b   c  d
        # out           (32, 100, 64)
        x = torch.einsum('abcd,abf->afd', [values, atten_weights])

        x = self.layer2(x)
        return x, atten_weights


def print_review(rev_text, sbs1, sbs2, lbl1, lbl2):
    """
    prints portion of the review (20-30 first words), with the sub-scores each work obtained
    prints also the final scores, the softmaxed prediction values and the true label values
    """
    # TODO: implement


def accuracy(y_pred, y_true):
    """
    calculates the accuracy given a prediction and true labels
    """
    size = y_true.shape[0]
    y_pred = torch.softmax(y_pred, dim=1)
    rounded_y_pred = torch.round(y_pred)
    agree = y_true == rounded_y_pred
    val = sum([1 for i in range(size) if agree[i][0] == agree[i][1] == True])
    percentage = 100 * float(val) / size
    return percentage


def choose_model():
    """
    chooses model given 4 optional implementation
    :return: chosen model instance
    """
    if run_recurrent:
        if use_RNN:
            chosen_model = ExRNN(input_size, output_size, hidden_size)
        else:
            chosen_model = ExGRU(input_size, output_size, hidden_size)
    else:
        if atten_size > 0:
            chosen_model = ExLRestSelfAtten(input_size, output_size, hidden_size)
        else:
            chosen_model = ExMLP(input_size, output_size, hidden_size)
    print("Using model: " + chosen_model.name())
    return chosen_model


def perform_step(labels, output, reviews):
    """
    this function provides an output based on the chosen model
    :param labels: the labels associated to every review
    :param output: the previous output
    :param reviews: vectors of reviews
    :return: a loss and an output
    """
    if run_recurrent:  # Recurrent nets (RNN/GRU)
        hidden_state = model.init_hidden(int(labels.shape[0]))
        for i in range(num_words):
            output, hidden_state = model(reviews[:, i, :], hidden_state)  # HIDE
    else:  # Token-wise networks (MLP / MLP + Atten.)
        if atten_size > 0:  # MLP + atten
            sub_score, atten_weights = model(reviews)
        else:  # MLP
            sub_score = model(reviews)
        output = torch.mean(sub_score, 1)
    loss = criterion(output, labels)  # cross-entropy loss
    if run_recurrent:
        return loss, output
    else:
        return loss, output, sub_score


# %% initializers
model = choose_model()
if reload_model:
    print("Reloading model")
    model.load_state_dict(torch.load(model.name() + ".pth"))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss, test_loss = 1.0, 1.0
train_accuracy_arr, test_accuracy_arr = [0], [0]
train_loss_arr, test_loss_arr = [0], [0]
train_output, test_output = 0, 0  # just as initialization
train_size, test_size = len(train_dataset), len(test_dataset)

# %% train/test process
sizes = [64, 96, 128]
for hidden_size in sizes:
    for epoch in range(num_epochs):
        cur_train_batch, cur_test_batch = 0, 0  # for printing progress per epoch
        train_epoch_acc, test_epoch_acc = 0, 0  # the accuracy both for the train data and the test data

        print(f"Epoch [{epoch + 1}/{num_epochs}]")

        # TRAIN
        for train_labels, train_reviews, train_reviews_text in train_dataset:  # train batch
            cur_train_batch += 1
            if cur_train_batch % 100 == 0: print(f"batch: [{cur_train_batch}/{train_size}]", end="\r")
            loss, train_output, sub_score = perform_step(train_labels, train_output, train_reviews)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = 0.9 * float(loss.detach()) + 0.1 * train_loss
            train_loss_arr.append(train_loss)
            train_epoch_acc += accuracy(train_output, train_labels)  # summing to finally average

        train_epoch_acc /= train_size  # normalizing to achieve average
        train_accuracy_arr.append(train_epoch_acc)
        # TEST
        for test_labels, test_reviews, test_reviews_text in test_dataset:  # test batch
            cur_test_batch += 1
            if cur_test_batch % 100 == 0: print(f"batch: [{cur_test_batch}/{test_size}]", end="\r")
            loss, test_output, sub_score = perform_step(test_labels, test_output, test_reviews)
            test_loss = 0.8 * float(loss.detach()) + 0.2 * test_loss
            test_loss_arr.append(test_loss)
            test_epoch_acc += accuracy(test_output, test_labels)

            # if not run_recurrent:
            #     nump_subs = sub_score.detach().numpy()
            #     labels = test_labels.detach().numpy()
            #     print_review(test_reviews_text[0], nump_subs[0, :, 0], nump_subs[0, :, 1], labels[0, 0], labels[0, 1])

        test_epoch_acc /= test_size
        test_accuracy_arr.append(test_epoch_acc)

        print(
            f"Train Loss: {train_loss:.4f}, "
            f"Train Accuracy: {train_epoch_acc:.4f}, "
            f"Test Loss: {test_loss:.4f}, "
            f"Test Accuracy: {test_epoch_acc:.4f}"
        )
    # plot
    plt.plot(test_accuracy_arr)
    # re-initializing
    train_accuracy_arr, test_accuracy_arr = [0], [0]
    train_loss_arr, test_loss_arr = [0], [0]

# displaying plot
title = f"Test accuracy [{model.name()}]"
plt.title(title)
plt.legend(sizes, title="hidden size")
plt.xlabel("Epochs")
plt.xlim([0, num_epochs])
plt.ylabel("Accuracy")
plt.grid()
plt.savefig(title)
plt.show()
