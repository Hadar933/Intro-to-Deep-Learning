import torch.nn
import torch.nn as nn
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import torchvision.datasets as datasets

# loading MNIST and padding 28x28 -> 32x32
pad_and_tensorize = transforms.Compose([transforms.Pad(2), transforms.PILToTensor()])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=pad_and_tensorize)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=pad_and_tensorize)
print("Loading Dataset Completed")
latent_dim = 10

# %% plotting an example
sample = 1020
image = torch.squeeze(mnist_train[sample][0])
plt.imshow(image, cmap='gray')
plt.title(f"the number {mnist_train[sample][1]} from MNIST with dimensions {image.shape[0]} x {image.shape[1]}")
plt.show()


# %%
class AutoEncoder(nn.Module):
    """
    this class represents the Auto Encoder model (AE) and uses convolution to encode the input
    to a latent space of dimension 'latent_dim'
    """

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.Encoder = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=5),
            torch.nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.Decoder = torch.nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=5),
            nn.ReLU()
        )

    def forward(self, im):
        im = self.Encoder(im)
        im = self.Decoder(im)
        return im


AE = AutoEncoder()
AE.forward(mnist_test[0][0])
