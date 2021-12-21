import torch.nn
import torch.nn as nn
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import torchvision.datasets as datasets
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 30
LATENT_DIM = 8  # can easily up-sample to 32
# loading MNIST and padding 28x28 -> 32x32
pad_and_tensorize = transforms.Compose([transforms.Pad(2), transforms.PILToTensor()])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=pad_and_tensorize)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=pad_and_tensorize)

train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=BATCH_SIZE)

print("#########################\nLoading Dataset Completed\n#########################")
# %%
sample = random.randint(0, len(mnist_test))  # for plotting the sample'th sample throughout the exercise


# %%
class AutoEncoder(nn.Module):
    """
    this class represents the Auto Encoder model (AE) and uses convolution to encode the input
    to a latent space of dimension 'latent_dim'
    """

    def __init__(self):
        super(AutoEncoder, self).__init__()
        # general purpose:
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()
        # encoder weights:
        self.enc_conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(5, 5), stride=(1, 1))
        self.enc_conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(5, 5), stride=(1, 1))
        self.enc_conv3 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=(5, 5), stride=(1, 1))

        self.enc_fully_connected = nn.Linear(LATENT_DIM, LATENT_DIM)
        # decoder weights:
        self.dec_conv1 = nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=(2, 2), stride=(2, 2))
        self.dec_conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(2, 2), stride=(2, 2))

    def forward(self, im):
        # (N,C,H,W):=(batch size, channels, height, weight)
        # encoding:
        # (N,1,32,32)->(N,4,28,28)->(N,16,24,24)->(N,16,12,12)->(N,4,8,8)
        encoded = self.activation(self.enc_conv1(im))
        encoded = self.activation(self.enc_conv2(encoded))
        encoded = self.pool(encoded)
        encoded = self.activation(self.enc_conv3(encoded))
        encoded = self.activation(self.enc_fully_connected(encoded))

        # decoding:
        # (N,4,8,8)->(N,16,16,16)->(N,1,32,32)
        # decoded = self.dec_fully_connected(encoded)
        decoded = self.activation(self.dec_conv1(encoded))
        decoded = self.dec_conv2(decoded)
        decoded = self.sigmoid(decoded)  # sigmoid to normalize the values in [0,1]

        return encoded, decoded


# %%
def plot_reconstructed(epoch, model):
    """
    plots an image from the test set, to see how the AE improves as the epochs increases
    """
    dec, enc = model(torch.unsqueeze(mnist_test[sample][0], 1).float())
    plt.subplot(3, 4, epoch + 2)
    plt.imshow(enc[0][0].detach().numpy(), cmap="gray")
    plt.title(f"epoch #{epoch}.")


# %% TRAIN - TEST iterations
num_epochs = 2 # TODO: change to 11
learning_rate = 0.001

AE = AutoEncoder()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(AE.parameters(), lr=learning_rate)

fig = plt.figure()  # plots the re-constucted example
fig.suptitle(f"Reconstruction of the digit {mnist_test[sample][1]} (the {sample}'th sample from the Test set)")
plt.subplot(3, 4, 1)
plt.imshow(torch.squeeze(mnist_test[sample][0]), cmap="gray")
plt.title(f"Original.")

train_loss, test_loss = 1.0, 1.0
train_accuracy_arr, test_accuracy_arr = [0], [0]
train_loss_arr, test_loss_arr = [1], [1]
train_output, test_output = 0, 0  # just as initialization
train_size, test_size = len(train_loader), len(test_loader)

for epoch in range(num_epochs):
    plot_reconstructed(epoch, AE)
    cur_train_batch, cur_test_batch = 0, 0  # for printing progress per epoch
    train_epoch_acc, test_epoch_acc = 0, 0  # the accuracy both for the train data and the test data

    print(f"Epoch [{epoch + 1}/{num_epochs}]")

    # TRAIN
    for train_batch in train_loader:  # train batch
        optimizer.zero_grad()
        train_images = train_batch[0] / 255  # we dont need the labels for now, also we normalize
        train_images = train_images.to(device)
        cur_train_batch += 1
        if cur_train_batch % 100 == 0: print(f"batch: [{cur_train_batch}/{train_size}]", end="\r")
        enc, dec = AE(train_images)
        loss = criterion(dec, train_images)
        loss.backward()
        optimizer.step()
        train_loss = 0.9 * float(loss.detach()) + 0.1 * train_loss
        train_loss_arr.append(train_loss)
        # train_epoch_acc += accuracy(train_output, train_labels)  # summing to finally average

    # train_epoch_acc /= train_size  # normalizing to achieve average
    # train_accuracy_arr.append(train_epoch_acc)

    # TEST
    with torch.no_grad():
        for test_batch in test_loader:  # test batch
            test_images = test_batch[0] / 255
            test_images = test_images.to(device)
            cur_test_batch += 1
            if cur_test_batch % 100 == 0: print(f"batch: [{cur_test_batch}/{test_size}]", end="\r")
            enc, dec = AE(test_images)
            loss = criterion(dec, test_images)
            test_loss = 0.8 * float(loss.detach()) + 0.2 * test_loss
            test_loss_arr.append(test_loss)
            # test_epoch_acc += accuracy(test_output, test_labels)

        # test_epoch_acc /= test_size
        # test_accuracy_arr.append(test_epoch_acc)

    print(
        f"Train Loss: {train_loss:.4f}, "
        # f"Train Accuracy: {train_epoch_acc:.4f}, "
        f"Test Loss: {test_loss:.4f}, "
        # f"Test Accuracy: {test_epoch_acc:.4f}"
    )
plt.show()
# %% plotting loss
plt.plot(train_loss_arr), plt.plot(test_loss_arr)
plt.title("Loss Plot - AutoEncoder"), plt.legend(["Train", "Test"]), plt.xlabel("Epochs"), plt.ylabel("Loss")
plt.xlim([0, num_epochs])
plt.grid()
plt.savefig("Loss Plot - AutoEncoder")
plt.show()
