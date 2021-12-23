import torch.nn
import torch.nn as nn
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import torchvision.datasets as datasets
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 30
LATENT_DIM = 8  # can easily up-sample to 32
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
        self.enc_conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=(1, 1))
        self.enc_conv2 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=(3, 3), padding=(1, 1))

        # self.enc_fully_connected = nn.Linear(LATENT_DIM, LATENT_DIM)
        # decoder weights:
        self.dec_conv1 = nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=(2, 2), stride=(2, 2))
        self.dec_conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(2, 2), stride=(2, 2))

    def forward(self, im):
        # encoding:
        im = self.activation(self.enc_conv1(im))
        im = self.pool(im)
        im = self.activation(self.enc_conv2(im))
        im = self.pool(im)

        # decoding:
        im = self.activation(self.dec_conv1(im))
        im = self.dec_conv2(im)
        im = self.sigmoid(im)  # sigmoid to normalize the values in [0,1]

        return im


# %%
def plot_reconstructed(epoch, model):
    """
    plots an image from the test set, to see how the AE improves as the epochs increases
    """
    out = model(torch.unsqueeze(mnist_test[sample][0] / 255, 1).float())
    plt.subplot(3, 4, epoch + 2)
    plt.imshow(out[0][0].detach().numpy(), cmap="gray")
    plt.title(f"epoch #{epoch}.")


# %% TRAIN - TEST iterations
num_epochs = 11
learning_rate = 0.001

AE = AutoEncoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(AE.parameters(), lr=learning_rate)

fig = plt.figure()  # plots the re-constucted example
fig.suptitle(f"Reconstruction of the digit {mnist_test[sample][1]} (the {sample}'th sample from the Test set)")
plt.subplot(3, 4, 1)
plt.imshow(torch.squeeze(mnist_test[sample][0]), cmap="gray")
plt.title(f"Original.")

train_loss_arr, test_loss_arr = [1], [1]
# train_accuracy_arr, test_accuracy_arr = [0], [0]
train_size, test_size = len(train_loader), len(test_loader)

for epoch in range(num_epochs):
    plot_reconstructed(epoch, AE)
    cur_train_batch, cur_test_batch = 0, 0  # for printing progress per epoch
    # train_epoch_acc, test_epoch_acc = 0, 0  # the accuracy both for the train data and the test data

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
            out = AE(test_images)
            loss = criterion(out, test_images)
            test_loss += loss.item()
            # test_epoch_acc += accuracy(test_output, test_labels)

        # test_epoch_acc /= test_size
        # test_accuracy_arr.append(test_epoch_acc)
    train_loss = train_loss / train_size
    test_loss = test_loss / test_size
    train_loss_arr.append(train_loss)
    test_loss_arr.append(test_loss)

    print(
        f"Train Loss: {train_loss:.4f}, "
        # f"Train Accuracy: {train_epoch_acc:.4f}, "
        f"Test Loss: {test_loss:.4f}, "
        # f"Test Accuracy: {test_epoch_acc:.4f}"
    )
plt.xticks([])
plt.yticks([])
plt.show()
# %% plotting loss
plt.plot(train_loss_arr), plt.plot(test_loss_arr)
plt.title("Loss Plot - AutoEncoder"), plt.legend(["Train", "Test"]), plt.xlabel("Epochs"), plt.ylabel("Loss")
plt.xlim([0, num_epochs])
plt.grid()
plt.savefig("Loss Plot - AutoEncoder")
plt.show()
# %% plotting reconstruction for various examples
test_iter = iter(test_loader)
images = test_iter.next()[0]
images = images[:6, :, :, :] / 255  # taking 6 examples and normalizing
out = AE(images)
images = torch.squeeze(images)
out = torch.squeeze(out)
im_to_plot = [images[i, :, :].detach().numpy() for i in range(6)]
out_to_plot = [out[i, :, :].detach().numpy() for i in range(6)]
fig = plt.figure()  # plots the re-constucted example
fig.suptitle(f"Reconstruction of 6 examples (from the Test set)")
for j in range(6):
    plt.subplot(2, 6, j+1)
    plt.imshow(im_to_plot[j],cmap="gray")
    plt.subplot(2, 6, j + 7)
    plt.imshow(out_to_plot[j],cmap="gray")
plt.savefig("Reconstruction of 6 examples (from the Test set)")
plt.show()
