import torch.nn
import torch.nn as nn
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import torchvision.datasets as datasets

BATCH_SIZE = 30
LATENT_DIM = 8

# loading MNIST and padding 28x28 -> 32x32
pad_and_tensorize = transforms.Compose([transforms.Pad(2), transforms.PILToTensor()])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=pad_and_tensorize)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=pad_and_tensorize)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=BATCH_SIZE)
print("#########################\nLoading Dataset Completed\n#########################")


# %% plotting an example
# sample = 1020
# image = torch.squeeze(mnist_train[sample][0])
# plt.imshow(image, cmap='gray')
# plt.title(f"the number {mnist_train[sample][1]} from MNIST with dimensions {image.shape[0]} x {image.shape[1]}")
# plt.show()


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
        self.dec_fully_connected = nn.Linear(LATENT_DIM, LATENT_DIM)

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


# AE = AutoEncoder()
# im = mnist_test[0][0][None, ...].float()
# out = AE.forward(im)

# %%
num_epochs = 10
learning_rate = 0.0001

AE = AutoEncoder()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(AE.parameters(), lr=learning_rate)

train_loss, test_loss = 1.0, 1.0
train_accuracy_arr, test_accuracy_arr = [0], [0]
train_loss_arr, test_loss_arr = [0], [0]
train_output, test_output = 0, 0  # just as initialization
train_size, test_size = len(mnist_train), len(mnist_test)
for epoch in range(num_epochs):
    cur_train_batch, cur_test_batch = 0, 0  # for printing progress per epoch
    train_epoch_acc, test_epoch_acc = 0, 0  # the accuracy both for the train data and the test data

    print(f"Epoch [{epoch + 1}/{num_epochs}]")

    # TRAIN
    for train_batch in train_loader:  # train batch
        optimizer.zero_grad()
        train_images = train_batch[0]/255  # we dont need the labels for now, also we normalize
        cur_train_batch += 1
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
    for test_batch in test_loader:  # test batch
        test_images = test_batch[0]/255
        cur_test_batch += 1
        if cur_test_batch % 100 == 0: print(f"batch: [{cur_test_batch}/{test_size}]", end="\r")
        enc, dec = AE(test_images)
        loss = criterion(dec, train_images)
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
