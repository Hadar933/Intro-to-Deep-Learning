import pandas as pd
import numpy as np

neg = pd.read_csv('Data/neg_A0201.txt', names=['seq'])
neg['label'] = -1
pos = pd.read_csv('Data/pos_A0201.txt', names=['seq'])
pos['label'] = 1

amino_letters = 'ARNDCEQGHILKMFPSTWYZ'  # Amino acids signs
one_hot_values = [str(i) + acid for i in range(9) for acid in amino_letters]


def generate_X(allell_df):
    allell_df[one_hot_values] = 0  # adding all one hot encoding labels
    for i, allell_name in enumerate(allell_df['seq']):
        allell_one_hot = [str(i) + acid for i, acid in enumerate(allell_name)]  # columns that should be assigned 1
        allell_df[allell_one_hot].iloc[i] = 1
    return allell_df


if __name__ == '__main__':
    X_pos = generate_X(pos)
