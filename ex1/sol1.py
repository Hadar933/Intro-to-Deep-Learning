import pandas as pd
import numpy as np
import torch.cuda
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import torch.utils.data as data_utils
import matplotlib.pyplot as plt

# global variables:
device = "cuda" if torch.cuda.is_available() else "cpu"
neg = pd.read_csv('Data/neg_A0201.txt', names=['seq'])
pos = pd.read_csv('Data/pos_A0201.txt', names=['seq'])
amino_letters = 'ACDEFGHIKLMNPQRSTVWY'  # Amino acids signs
one_hot_values = [str(i) + acid for i in range(9) for acid in amino_letters]  # one hot features


def generate_X(peptid_df):
    """
    generates a sample matrix with all the relevant dummy variables from the given peptide dataframe
    :param peptid_df: a dataframe with the peptids
    :return: altered df
    """
    peptid_df[one_hot_values] = 0  # adding all one hot encoding labels
    for i, peptid_name in enumerate(peptid_df['seq']):
        peptid_one_hot = [str(i) + acid for i, acid in enumerate(peptid_name)]  # columns that should be assigned 1
        peptid_df.at[i, peptid_one_hot] = 1
        if i % 100 == 0:
            print(f"pre-processing: completed {i}")
    return peptid_df.drop(["seq"], axis=1)


def preprocess():
    """
    :return: train and test X,y
    """
    X_pos, X_neg = generate_X(pos), generate_X(neg)
    y_pos, y_neg = np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])
    X, y = pd.concat([X_pos, X_neg]), np.concatenate([y_pos, y_neg])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    X_train_tensor = torch.tensor(X_train.values).float()
    X_test_tensor = torch.tensor(X_test.values).float()
    y_train_tensor = torch.tensor(y_train).float()
    y_test_tensor = torch.tensor(y_test).float()

    train = data_utils.TensorDataset(X_train_tensor, y_train_tensor)
    test = data_utils.TensorDataset(X_test_tensor, y_test_tensor)
    return train, test, y_test


class MLP(nn.Module):
    """
    Multilayer Perceptron.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(180, 360),
            nn.ReLU(),
            nn.Linear(360, 360),
            nn.ReLU(),
            nn.Linear(360, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        """
        forward pass
        :param x: network input
        :return: network output
        """
        return self.layers(x)


def accuracy(y_pred, y_true):
    """
    calculates the accuracy given a prediction and true labels
    """
    rounded_y_pred = torch.round(torch.sigmoid(y_pred))
    agree = (y_true == rounded_y_pred).sum()
    return 100 * agree.float() / y_true.shape[0]


def train_iteration(dataloader, model, loss_fn, optimizer, train_loss_arr):
    """
    trains the network using SGD (1 epoch)
    """
    size = len(dataloader.dataset)
    model.train()

    epoch_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        y = y[..., np.newaxis]
        prediction = model(X)
        loss = loss_fn(prediction, y)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    train_loss_arr.append(epoch_loss / size)
    print(f"Train Loss: {epoch_loss / size :.5f};")


def test_iteration(dataloader, model, loss_fn, test_loss_arr):
    """
    tests the trained model and provides train loss as well as a prediction
    for the test data
    """
    size = len(dataloader.dataset)
    all_predictions = []
    model.eval()  # tell pyTorch we do not use back-prop

    epoch_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y = y[..., np.newaxis]
            prediction = model(X)
            prediction = torch.round(torch.sigmoid(prediction))
            as_lst = [item.item() for item in prediction]
            all_predictions += as_lst
            loss = loss_fn(prediction, y)
            epoch_loss += loss.item()

        test_loss_arr.append(epoch_loss / size)
        print(f"Test Loss: {epoch_loss / size :.5f};")

    return all_predictions


def measurements_plots(y, y_pred):
    """
    plots a confusion matrix, and prints relevant classification reports
    :param y: true labels
    :param y_pred: predicted labels
    """
    cm = confusion_matrix(y, y_pred)
    display_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Neg", "Pos"])
    display_cm.plot()
    plt.title("confusion matrix")
    plt.savefig("confusion_mat_fig.png", format='png')
    plt.show()
    print(classification_report(y, y_pred))


# %% preprocessing the data
train_data, test_data, y_test = preprocess()

# %% initializing parameters

epochs, batch_size, learning_rate = 20, 64, 7e-3
model = MLP().to(device)
pos_weight_ratio = torch.ones(1) * (neg.shape[0] / pos.shape[0])
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_ratio)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
train_dataloader = DataLoader(train_data, batch_size)
test_dataloader = DataLoader(test_data, batch_size)

# %% training
test_loss_arr, train_loss_arr = [], []
y_pred = 0
for t in range(epochs):
    print(f"Epoch: {t}")
    train_iteration(train_dataloader, model, loss_fn, optimizer, train_loss_arr)
    y_pred = test_iteration(test_dataloader, model, loss_fn, test_loss_arr)

# final_pred = np.array([pred[0] for sublist in y_pred for pred in sublist])

# %% plotting train and test error
plt.plot(test_loss_arr)
plt.plot(train_loss_arr)
plt.title(f"Train\Test error (epochs={epochs}, batch size={batch_size}, learning rate={learning_rate})")
plt.xlabel("Epochs")
plt.ylabel("Error (Arb. Units)")
plt.legend(["Test", "Train"])
plt.grid()
plt.savefig("test_train_loss_fig.png", format='png')
plt.show()

# %%
measurements_plots(y_test, y_pred)


# %% for question 6,7

def predict_peptide_from_spark(spark):
    """
    :param spark: a string that represents the 1273-amino-acid sequence
    of the spark protein in SARS-CoV-2
    :return: top 5 predicted peptides
    """
    peptide_len = 9
    split_spark = [spark[i:i + 9] for i in range(0, len(spark) - (peptide_len - 1), 1)]
    spark_df = pd.DataFrame({'seq': split_spark})
    X_spark = generate_X(spark_df)
    X_spark_tensor = torch.tensor(X_spark.values).float()
    prediction = model(X_spark_tensor)
    prediction = torch.sigmoid(prediction)
    pred_and_peptide = [(spark_df['seq'][i], prediction[i].item()) for i in range(len(spark_df['seq']))]
    pred_and_peptide.sort(key=lambda x: x[1])
    top_5 = [item[0] for item in pred_and_peptide[-5:]]
    return top_5


# %%

spark = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKS" \
        "NIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKN" \
        "IDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALD" \
        "PLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCF" \
        "TNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYF" \
        "PLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDIT" \
        "PCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARS" \
        "VASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQ" \
        "VKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTI" \
        "TSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDI" \
        "LSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTA" \
        "PAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASV" \
        "VNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
print(f"top five peptides: {predict_peptide_from_spark(spark)}")


# %%
def optimize_sequence(model):
    """
    find the best fitting peptide, when initialized as a random input
    :param model:
    :return:
    """
    x = torch.randn((1, 180), device=device, dtype=torch.float, requires_grad=True)
    loss_arr = []
    optimizer = torch.optim.SGD([x], lr=learning_rate)
    model.requires_grad_(False)
    y = torch.tensor([1]).to(device)
    y = y.unsqueeze(1).float()
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    num_iters = 100
    for i in range(num_iters):
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_fn(prediction, y)
        loss_arr.append(loss.item())
        loss.backward()
        optimizer.step()

    x = torch.sigmoid(x)
    # after optimizing x, we need to convert it to a peptide representation.
    # we will take the maximum index for every 20 consecutive values
    consecutive_20s = [x.data[0][i:i + 20] for i in range(0, len(x.data[0]), 20)]
    max_indexes = [torch.argmax(item).item() for item in consecutive_20s]
    final_peptide = "".join([amino_letters[ind] for ind in max_indexes])
    return final_peptide, loss_arr


# %%
peptide, loss_arr = optimize_sequence(model)
plt.plot(loss_arr)
plt.title(f"Optimizing sequence loss (result peptide: {peptide})")
plt.xlabel("Iterations")
plt.ylabel("Error (Arb. units")
plt.grid()
plt.savefig("optimize_seq_plot.png", format='png')
plt.show()


# %%
def shift(s):
    first_s = s
    while s != first_s:
        s0 = s[0]
        s = s[1:]
        s = s + s0
        print(s)

