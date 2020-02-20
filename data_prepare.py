import numpy as np
import pandas as pd

def removeEmptyRows(data):
    n_samples, n_days = data.shape[0], data.shape[1]
    
    rows = []
    for n in range(n_samples):
        if np.isnan(data[n,0]):
            continue
        else: rows.append(data[n].reshape(1,-1))

    data = np.vstack(rows)

    return data

import argparse
parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--data_dir", type=str, default='')
parser.add_argument("--train_file", type=str, default='')
parser.add_argument("--dummy_file", type=str, default='')
parser.add_argument("--tr_out_file", type=str, default='')
parser.add_argument("--val_out_file", type=str, default='')
parser.add_argument("--test_out_file", type=str, default='')

args = parser.parse_args()

if __name__ == "__main__":
    train_file = args.data_dir+'/'+args.train_file
    dummy_file = args.data_dir+'/'+args.dummy_file
    
    tr_out_file = args.data_dir+'/'+args.tr_out_file
    val_out_file = args.data_dir+'/'+args.val_out_file
    test_out_file = args.data_dir+'/'+args.test_out_file

    df_train = pd.read_csv(train_file)
    df_dummy = pd.read_csv(dummy_file)

    np_train = np.asarray(df_train)
    np_dummy = np.asarray(df_dummy)

    np_train = removeEmptyRows(np_train)
    np_dummy = removeEmptyRows(np_dummy)

    ones = np.ones((np_train.shape[0],1))
    zeros = np.zeros((np_dummy.shape[0],1))

    # concatconcat
    np_data = np.vstack([np_train, np_dummy])
    np_labels = np.vstack([ones, zeros])
    
    # shuffle
    ids = list(range(np_data.shape[0]))
    np.random.seed(11)
    np.random.shuffle(ids)

    np_data = np_data[ids]
    np_labels = np_labels[ids]
    
    # split into train,valid,test
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 1 - train_ratio - valid_ratio

    n_samples = np_data.shape[0]
    train_split = int(np.ceil(n_samples * train_ratio))
    valid_split = int(np.ceil(n_samples * (train_ratio + valid_ratio)))

    train_data, train_labels = np_data[:train_split], np_labels[:train_split]
    valid_data, valid_labels = np_data[train_split:valid_split], np_labels[train_split:valid_split]
    test_data, test_labels = np_data[valid_split:], np_labels[valid_split:]

    train_data = np.hstack([train_data, train_labels])
    valid_data = np.hstack([valid_data, valid_labels])
    test_data = np.hstack([test_data, test_labels])

    with open(tr_out_file, "wt") as fp:
        np.savetxt(fp, train_data, delimiter=',')
    with open(val_out_file, "wt") as fp:
        np.savetxt(fp, valid_data, delimiter=',')
    with open(test_out_file, "wt") as fp:
        np.savetxt(fp, test_data, delimiter=',')
