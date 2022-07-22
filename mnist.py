# Code modified from: https://github.com/hsjeong5/MNIST-for-Numpy

import numpy as np
import pandas as pd
from urllib import request
import gzip
import pickle
import os
import pathlib

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]
SAVE_PATH = pathlib.Path("data/original_mnist")
SAVE_PATH_EMNIST = pathlib.Path("data/emnist")

def download_mnist():
    SAVE_PATH.mkdir(exist_ok=True, parents=True)
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        filepath = SAVE_PATH.joinpath(name[1])
        if filepath.is_file():
            continue
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], filepath)

def extract_emnist():
    emnist = {}
    balanced_train = pd.read_csv('data/emnist/emnist-balanced-train.csv')
    balanced_test = pd.read_csv('data/emnist/emnist-balanced-test.csv')

    SAVE_PATH_EMNIST.mkdir(exist_ok=True, parents=True)
    save_path = SAVE_PATH_EMNIST.joinpath("emnist.pkl")
        
    def convert_to_2d(df):
        return df.reshape([28, 28])
    
    balanced_train_data = balanced_train.iloc[:, 1:].values
    balanced_test_data = balanced_test.iloc[:, 1:].values

    emnist['training_images'] = np.array(list(map(convert_to_2d, balanced_train_data)))
    emnist['test_images'] = np.array(list(map(convert_to_2d, balanced_test_data)))
    emnist['training_labels'] = balanced_train.iloc[:, 0].values
    emnist['test_labels'] = balanced_test.iloc[:, 0].values

    with open(save_path, 'wb') as f:
        pickle.dump(emnist, f)

def extract_mnist():
    save_path = SAVE_PATH.joinpath("mnist.pkl")
    if save_path.is_file():
        return
    mnist = {}
    # Load images
    for name in filename[:2]:
        path = SAVE_PATH.joinpath(name[1])
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            print(data.shape)
            mnist[name[0]] = data.reshape(-1,   28*28)
    # Load labels
    for name in filename[2:]:
        path = SAVE_PATH.joinpath(name[1])
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
            mnist[name[0]] = data
    with open(save_path, 'wb') as f:
        pickle.dump(mnist, f)

def load():
    download_mnist()
    extract_mnist()
    dataset_path = SAVE_PATH.joinpath("mnist.pkl")
    with open(dataset_path, 'rb') as f:
        mnist = pickle.load(f)
    X_train, Y_train, X_test, Y_test = mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]
    return X_train.reshape(-1, 28, 28), Y_train, X_test.reshape(-1, 28, 28), Y_test

def load_emnist():
    extract_emnist()
    dataset_path = SAVE_PATH_EMNIST.joinpath("emnist.pkl")
    with open(dataset_path, 'rb') as f:
        emnist = pickle.load(f)
    X_train, Y_train, X_test, Y_test = emnist["training_images"], emnist["training_labels"], emnist["test_images"], emnist["test_labels"]
    return X_train.reshape(-1, 28, 28), Y_train, X_test.reshape(-1, 28, 28), Y_test


if __name__ == '__main__':
    init()