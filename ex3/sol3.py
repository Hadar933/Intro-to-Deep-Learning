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
