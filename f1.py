from google.colab import drive
drive.mount('/content/drive')

import cv2
import os
import numpy as np
import pandas as pd

def split(n, rr, filenames, labels):
    seed = 7
    np.random.seed(seed)
    indices = np.random.choice(n, rr, replace = False)
    # print(indices)
    A_X, A_Y, B_X, B_Y = [], [], [], []
    for i in range(n):
        if i in indices:
            A_X.append(filenames[i])
            A_Y.append(labels[i])
        else:
            B_X.append(filenames[i])
            B_Y.append(labels[i])
    return A_X, A_Y, B_X, B_Y

current = r'/content/drive/My Drive/gray_img/gray_img'
df = pd.read_csv(r'/content/drive/My Drive/gt.csv', sep = ',', header = None)
labels = df.values.tolist()
filenames = []
for i in range(1, 1330):
    filenames.append(os.path.join(current, str(i) + '.png'))
trainAval_img, trainAval_labels, test_img, test_labels = split(1229, 800, filenames, labels)
train_img, train_labels, val_img, val_labels = split(800, 600, trainAval_img, trainAval_labels)
#print(len(train_img), len(test_img), len(val_img))

train_Y = np.array(train_labels)
val_Y = np.array(val_labels)
test_Y = np.array(test_labels)

tr_X, val_X, te_X = [], [], []
for name in filenames:
    image = cv2.imread(name)
    if name in train_img:
        tr_X.append(image)
    elif name in val_img:
        val_X.append(image)
    elif name in test_img:
        te_X.append(image)    
train_X_np = np.array(tr_X)
val_X_np = np.array(val_X)
test_X_np = np.array(te_X)
print(train_X_np.shape, val_X_np.shape, test_X_np.shape)

train_set_X, val_set_X, test_set_X = [], [], []
dim = (300, 300)
for ii in train_X_np:
    grey = cv2.cvtColor(ii, cv2.COLOR_BGR2GRAY)
    grey_resized = cv2.resize(grey, dim)
    train_set_X.append(grey_resized)
train_X = np.array(train_set_X)
for ii in val_X_np:
    grey = cv2.cvtColor(ii, cv2.COLOR_BGR2GRAY)
    grey_resized = cv2.resize(grey, dim)
    val_set_X.append(grey_resized)
val_X = np.array(val_set_X)
for ii in test_X_np:
    grey = cv2.cvtColor(ii, cv2.COLOR_BGR2GRAY)
    grey_resized = cv2.resize(grey, dim)
    test_set_X.append(grey_resized)
test_X = np.array(test_set_X)
