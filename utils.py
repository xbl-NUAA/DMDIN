import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import  roc_auc_score
from torch.utils.data import DataLoader
from scipy.io import loadmat
from sklearn.model_selection import KFold
import sys

def processdata(path, fold):
    data = np.load(path)
    nums, numv = np.shape(data)
    fold_num = int(nums/fold)

    x_train_list = []
    y_train_list = []
    x_val_list = []
    y_val_list = []
    x_test_list = []
    y_test_list = []

    X, Y = data[:, : numv - 1], data[:, numv - 1:]

    KF = KFold(n_splits=fold)

    for train_index, test_index in KF.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        X_val = X_train[:fold_num,:]
        Y_val = Y_train[:fold_num,:]
        X_train = X_train[fold_num:,:]
        Y_train = Y_train[fold_num:,:]


        min_max_scaler = preprocessing.MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)  # 归一化
        X_val = min_max_scaler.fit_transform(X_val)  # 归一化
        X_test = min_max_scaler.fit_transform(X_test)  # 归一化


        XY = np.hstack((X_train, Y_train))
        sort_XY = np.array(sorted(XY, key=lambda x: x[XY.shape[1] - 1]))
        X_train = sort_XY[:, :sort_XY.shape[1] - 1]
        Y_train = sort_XY[:, sort_XY.shape[1] - 1:]

        x_train_list.append(X_train)
        x_val_list.append(X_val)
        x_test_list.append(X_test)
        y_train_list.append(Y_train)
        y_val_list.append(Y_val)
        y_test_list.append(Y_test)

    return x_train_list, y_train_list, x_val_list, y_val_list, x_test_list, y_test_list


def getfold(X_train,Y_train,X_val,Y_val,X_test,Y_test):
    xy_tr = list(zip(X_train, Y_train))
    xy_val = list(zip(X_val, Y_val))
    xy_te = list(zip(X_test, Y_test))

    class_list = list()
    y_list = Y_train.squeeze().tolist()
    labels = np.unique(Y_train.squeeze())
    for i in range(len(labels)):
        class_list.append(y_list.count(i))
    return xy_tr, xy_val, xy_te, class_list

def loaddata(x_train_list,y_train_list,x_val_list,y_val_list,x_test_list,y_test_list,i):
    xy_tr, xy_val, xy_te, class_list = getfold(x_train_list[i],y_train_list[i],x_val_list[i],y_val_list[i],x_test_list[i],y_test_list[i])

    batch_size  = class_list[0] + class_list[1]
    train_loader = DataLoader(
        xy_tr, batch_size=batch_size)

    val_loader = DataLoader(
        xy_val, batch_size=batch_size)

    test_loader = DataLoader(
        xy_te, batch_size=batch_size)

    return train_loader, val_loader, test_loader, class_list

def list_to_np(x):
    temp = np.zeros((1, x[0].shape[1]))
    for i in range(len(x)):
        temp = np.vstack((temp,x[i].data.cpu().numpy()))
    return temp[1:,:]

def divide(x, modalities):
    x_list = []
    for i in range(len(modalities) - 1):
        index1 = modalities[i]
        index2 = modalities[i + 1]
        x_list.append(x[:,index1:index2])

    return x_list

def eval(z,z_te,svmc = 0.5):
    z = list_to_np(z)
    z_te = list_to_np(z_te)
    x_train, y_train = z[:,:-1], z[:,-1]

    x_test, y_test = z_te[:, :-1], z_te[:, -1]

    min_max_scaler = preprocessing.MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x_train)
    x_test = min_max_scaler.fit_transform(x_test)

    clf = SVC(kernel='linear', C=svmc)
    clf.fit(x_train, y_train.ravel())
    acc_tr = clf.score(x_train, y_train)
    acc_te = clf.score(x_test, y_test)

    auc = roc_auc_score(y_test, clf.decision_function(x_test))

    y_true = y_test.squeeze()
    y_pre = clf.predict(x_test)
    fptn = 0
    tn = 0
    tpfn = 0
    tp = 0
    for n in range(len(y_pre)):
        if y_true[n] == 0:
            fptn = fptn + 1
            if y_pre[n] == y_true[n]:
                tn = tn + 1
        if y_true[n] == 1:
            tpfn = tpfn + 1
            if y_pre[n] == y_true[n]:
                tp = tp + 1
    if tn == 0:
        spe = 0
    else:
        spe = tn / fptn
    if tp == 0:
        sen = 0
    else:
        sen = tp / tpfn
    return acc_tr,acc_te,sen,spe,auc