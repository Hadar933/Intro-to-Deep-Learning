# %% imports
import torch.nn
import torch.nn as nn
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import torchvision.datasets as datasets
import random
from matplotlib.pyplot import figure
from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score

# %% loading data
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 30
pad_and_tensorize = transforms.Compose([transforms.Pad(2), transforms.PILToTensor()])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=pad_and_tensorize)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=pad_and_tensorize)

train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=BATCH_SIZE)

print("Loading Dataset Completed")
# %% initializing a random sample
RAND_SAMPLE = random.randint(0, len(mnist_test))  # for plotting the sample'th sample throughout the exercise


# %% Auto Encoder class
class AutoEncoder(nn.Module):
    """
    this class represents the Auto Encoder model (AE) and uses convolution to encode the input
    to a latent space of dimension 'latent_dim'
    """

    def __init__(self, latent_dim):
        super(AutoEncoder, self).__init__()
        # general purpose:
        self.test_loss_arr = []
        self.train_loss_arr = []
        self.latent_dim = latent_dim
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()
        # encoder weights: (when padding=1 with kernel=3 the shape stays the same)
        self.enc_conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=(1, 1))
        self.enc_conv2 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=(3, 3), padding=(1, 1))

        self.enc_fully_connected = nn.Linear(8, latent_dim)  # we change latent dim to compare results
        # decoder weights:
        self.dec_fully_connected = nn.Linear(latent_dim, 8)
        self.dec_conv1 = nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=(2, 2), stride=(2, 2))
        self.dec_conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(2, 2), stride=(2, 2))

    def forward(self, im):
        # encoding:
        im = self.activation(self.enc_conv1(im))
        im = self.pool(im)
        im = self.activation(self.enc_conv2(im))
        im = self.pool(im)
        im = self.enc_fully_connected(im)

        # decoding:
        im = self.dec_fully_connected(im)
        im = self.activation(self.dec_conv1(im))
        im = self.dec_conv2(im)
        im = self.sigmoid(im)  # sigmoid to normalize the values in [0,1]
        return im


# %% Initializers
num_epochs = 20
learning_rate = 0.001
# models = [AutoEncoder(d) for d in range(1, 9)]
models = [AutoEncoder(2)]
criterion = nn.MSELoss()
# %% test train iterations
train_loss_arr, test_loss_arr = [], []
train_size, test_size = len(train_loader), len(test_loader)
for AE in models:
    print(f"##### AE with latent dim = {8 * AE.latent_dim} #####")
    optimizer = torch.optim.Adam(AE.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        cur_train_batch, cur_test_batch = 0, 0  # for printing progress per epoch

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        train_loss, test_loss = 0, 0
        # TRAIN
        for train_batch in train_loader:  # train batch
            optimizer.zero_grad()
            train_images = train_batch[0] / 255  # we dont need the labels for now, also we normalize
            train_images = train_images.to(device)
            cur_train_batch += 1
            if cur_train_batch % 100 == 0: print(f"batch: [{cur_train_batch}/{train_size}]", end="\r")
            out = AE(train_images)
            loss = criterion(out, train_images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # TEST
        with torch.no_grad():
            for test_batch in test_loader:  # test batch
                test_images = test_batch[0] / 255
                test_images = test_images.to(device)
                cur_test_batch += 1
                if cur_test_batch % 100 == 0: print(f"batch: [{cur_test_batch}/{test_size}]", end="\r")
                out = AE(test_images)
                loss = criterion(out, test_images)
                test_loss += loss.item()

        train_loss = train_loss / train_size
        test_loss = test_loss / test_size
        AE.train_loss_arr.append(train_loss)
        AE.test_loss_arr.append(test_loss)

        print(
            f"Train Loss: {train_loss:.4f}, "
            f"Test Loss: {test_loss:.4f}, "
        )
# %% plotting loss
for AE in models:
    plt.plot(AE.train_loss_arr)
plt.title("Train Loss for various AEs"), plt.xlabel("Epochs"), plt.ylabel("Loss")
plt.legend([f"FC 8x{AE.latent_dim}" for AE in models], loc="upper right")
plt.xlim([0, 10])
plt.grid()
plt.savefig("Train Loss - AutoEncoder")
plt.show()
for AE in models:
    plt.plot(AE.test_loss_arr)
plt.title("Test Loss for various AEs"), plt.xlabel("Epochs"), plt.ylabel("Loss")
plt.legend([f"FC 8x{AE.latent_dim}" for AE in models], loc="upper right")
plt.xlim([0, 10])
plt.grid()
# plt.savefig("Test Loss - AutoEncoder")
plt.show()
# %% visual comparison of a single sample

im = torch.unsqueeze(mnist_test[RAND_SAMPLE][0] / 255, 1).float()
fig, axs = plt.subplots(1, 9)
plt.figure(figsize=(8, 2), dpi=80)
axs[0].imshow(im[0][0].detach().numpy() / 255, cmap="gray")
axs[0].set_title("original")
for pos, AE in enumerate(models):
    y_pred = AE(torch.unsqueeze(mnist_test[RAND_SAMPLE][0] / 255, 1).float())
    axs[pos + 1].imshow(y_pred[0][0].detach().numpy(), cmap="gray")
    axs[pos + 1].set_title(f"8x{AE.latent_dim}")
fig.savefig("Original vs Reconstructed image for various AEs")
plt.show()

# %% plotting reconstruction for various examples
test_iter = iter(test_loader)
images = test_iter.next()[0]
images = images[:6, :, :, :] / 255  # taking 6 examples and normalizing
y_pred = AE(images)
images = torch.squeeze(images)
y_pred = torch.squeeze(y_pred)
im_to_plot = [images[i, :, :].detach().numpy() for i in range(6)]
out_to_plot = [y_pred[i, :, :].detach().numpy() for i in range(6)]
fig = plt.figure()  # plots the re-constucted example
fig.suptitle(f"Reconstruction of 6 examples (from the Test set)")
for j in range(6):
    plt.subplot(2, 6, j + 1)
    plt.imshow(im_to_plot[j], cmap="gray")
    plt.subplot(2, 6, j + 7)
    plt.imshow(out_to_plot[j], cmap="gray")
# plt.savefig("Reconstruction of 6 examples (from the Test set)")
plt.show()
# %% using SVM as a basic classifier, we compare the accuracy rates for every model
classifier = svm.SVC(decision_function_shape='ovo')
padder = transforms.Pad(2)
train_size = 4200  # using only a fraction of the actual train size, as this is sufficient in the training process
test_size = train_size // 6  # dividing by 6 to remain in the original train/test ratio
accuracies = []
X_train = padder(torch.unsqueeze(mnist_train.data, 1)) / 255
X_train = X_train[:train_size, :, :, :]
X_test = padder(torch.unsqueeze(mnist_test.data, 1)) / 255
X_test = X_test[:test_size, :, :, :]
y_train = mnist_train.train_labels.numpy()
y_train = y_train[:train_size]
y_test = mnist_test.test_labels.numpy()
y_test = y_test[:test_size]
for AE in models:
    print(f"AE with 8x{AE.latent_dim} latent dim...")
    decoded_train = torch.squeeze(AE(X_train)).detach().numpy()
    decoded_test = torch.squeeze(AE(X_test)).detach().numpy()
    # decoded_train = torch.squeeze(X_train).detach().numpy()
    # decoded_test = torch.squeeze(X_test).detach().numpy()
    decoded_train = decoded_train.reshape(decoded_train.shape[0], decoded_train.shape[1] * decoded_train.shape[2])
    decoded_test = decoded_test.reshape(decoded_test.shape[0], decoded_test.shape[1] * decoded_test.shape[2])

    classifier.fit(decoded_train, y_train)
    y_pred = classifier.predict(decoded_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(acc)

print(accuracies)
# %% plotting the SVM accuracy rates
acc_vals = accuracies
x_vals = ["8x1", "8x2", "8x3", "8x4", "8x6", "8x7"]
plt.grid(color='gray', linestyle='dashed')
plt.bar(x_vals, acc_vals, width=0.5, color=["rosybrown", "lightcoral", "indianred", "brown", "firebrick", "maroon"])
plt.ylim([0.84, 0.933])
plt.xlabel("Dimension of FC layer (latent dim)")
plt.ylabel("Accuracy")
plt.title("SVM Accuracy as a function of the latent dimension")
plt.savefig("Accuracy as a function of the latent dimension")
plt.show()
# %% Pearson correlation
X_test = padder(torch.unsqueeze(mnist_test.data, 1)) / 255
pearson_arr = []
for trained_AE_model in models:
    print(trained_AE_model.latent_dim)
    encoder = torch.nn.Sequential(
        trained_AE_model.enc_conv1,
        trained_AE_model.activation,
        trained_AE_model.pool,
        trained_AE_model.enc_conv2,
        trained_AE_model.activation,
        trained_AE_model.pool,
        trained_AE_model.enc_fully_connected
    )
    encoded_test = encoder(X_test)
    encoded_test = torch.flatten(encoded_test, 1)
    print(encoded_test)
    pearson = torch.corrcoef(encoded_test.T)
    total_pearson = torch.mean(pearson) / 2  # dividing by 2 because pearson is symmetric
    pearson_arr.append(total_pearson.item())
# %% plotting pearson vs dim:
x_data = ["8x3", "8x4", "8x5", "8x6", "8x8"]
y_data = [0.0184016, 0.0052582, 0.0037022, 0.0030080, 0.0004629]
fig, ax = plt.subplots()
ax.plot(x_data, y_data)
for i, txt in enumerate(y_data):
    ax.annotate(f"{100 * txt:.4}", (i, y_data[i]))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.title("Pearson decay for increasingly larger latent dim")
plt.xlim([0, 4])
plt.ylim([0, 0.019])
plt.grid()
plt.savefig("Pearson decay for increasingly larger latent dim")
plt.show()

# %% Transfer Learning - first part : training only the MLP addition
pretrained_AE = models[0]
fc1_out_dim, fc2_out_dim = 100, 100  # to experiment with


class Transfer(nn.Module):
    def __init__(self, train_encoder):
        super(Transfer, self).__init__()
        self.train_loss_arr = []
        self.test_loss_arr = []
        self.train_encoder = train_encoder
        # utility layers:
        self.flat = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)
        self.activation = nn.ReLU()
        self.pool = pretrained_AE.pool
        # decoder pre-trained layers:

        # additional MLP layers:
        self.fc1 = nn.Linear(64, fc1_out_dim)
        self.fc2 = nn.Linear(fc1_out_dim, fc2_out_dim)
        self.fc3 = nn.Linear(fc2_out_dim, 10)  # 10 output neurons as for 10 classes of MNIST
        if not train_encoder:
            self.enc_conv1 = pretrained_AE.enc_conv1
            self.enc_conv2 = pretrained_AE.enc_conv2
            self.enc_fc = pretrained_AE.enc_fully_connected
            self.enc_conv1.requires_grad_(False)
            self.enc_conv2.requires_grad_(False)
            self.enc_fc.requires_grad_(False)
        else:  # training encoder from the beginning
            self.enc_conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=(1, 1))
            self.enc_conv2 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=(3, 3), padding=(1, 1))
            self.enc_fc = nn.Linear(8, 2)

    def forward(self, im):
        # encoding:
        im = self.activation(self.enc_conv1(im))
        im = self.pool(im)
        im = self.activation(self.enc_conv2(im))
        im = self.pool(im)
        im = self.enc_fc(im)

        # passing through MLP:
        im = self.flat(im)
        im = self.activation(self.fc1(im))
        im = self.activation(self.fc2(im))
        im = self.fc3(im)
        im = self.softmax(im)
        return im


# %% train test iteration for the Transform model using SMALL train size
two_models = [Transfer(True), Transfer(False)]
for Transfer_model in two_models:
    print(f"Training Encoder?: {Transfer_model.train_encoder}.")
    train_loss_arr, test_loss_arr = [], []
    train_size, test_size = len(train_loader), len(test_loader)
    optimizer = torch.optim.Adam(Transfer_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        cur_train_batch, cur_test_batch = 0, 0  # for printing progress per epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        train_loss, test_loss = 0, 0
        # TRAIN (based on a small portion of the train data)
        batches_used, batches_to_use = 0, 10  # we will use this to only consider small number of train
        for train_batch in train_loader:  # train batch
            print(f"batch: [{batches_used}/{batches_to_use}]", end="\r")
            if batches_used == 10:
                break
            else:
                batches_used += 1
            X_train, y_train = train_batch
            optimizer.zero_grad()
            X_train = X_train.float() / 255
            y_pred = Transfer_model(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # TEST (based on all the test data)
        with torch.no_grad():
            for X_test, y_test in test_loader:  # test batch
                X_test = X_test.float() / 255
                cur_test_batch += 1
                if cur_test_batch % 100 == 0: print(f"batch: [{cur_test_batch}/{test_size}]", end="\r")
                y_pred = Transfer_model(X_test)
                loss = criterion(y_pred, y_test)
                test_loss += loss.item()

        train_loss = train_loss / batches_to_use
        test_loss = test_loss / test_size
        Transfer_model.train_loss_arr.append(train_loss)
        Transfer_model.test_loss_arr.append(test_loss)

        print(
            f"Train Loss: {train_loss:.4f}, "
            f"Test Loss: {test_loss:.4f}, "
        )
# %% Plotting train and test loss of the Transformer:
model = two_models[0]
plt.plot(model.train_loss_arr)
plt.plot(model.test_loss_arr)
cond = '(Freezed)' if model.train_encoder else '(Trained)'
plt.title(f"Train/Test Loss - {cond} Encoder weights")
plt.ylabel("Loss"), plt.xlabel("Epochs")
plt.legend([f"Train {cond}", f"Test {cond}"])
plt.grid()
plt.savefig(f"Train-Test Loss - {cond} Encoder weights")
plt.show()
# %% GAN implementation
