import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import  roc_auc_score
from torch.utils.data import DataLoader



def getfold(path, fold, i):
    X_train, Y_train = None,  # load i-th fold data
    X_val, Y_val = None,None
    X_test, Y_test = None,None
    xy_tr = list(zip(X_train, Y_train))
    xy_val = list(zip(X_val, Y_val))
    xy_te = list(zip(X_test, Y_test))

    class_list = list()
    y_list = Y_train.squeeze().tolist()
    labels = np.unique(Y_train.squeeze())
    for i in range(len(labels)):
        class_list.append(y_list.count(i))
    return xy_tr, xy_val, xy_te, class_list

def loaddata(path, fold,i, batch_size = 32):
    xy_tr, xy_val, xy_te = getfold(path, fold, i)

    train_loader = DataLoader(
        xy_tr, shuffle=False, batch_size=batch_size)

    val_loader = DataLoader(
        xy_val, shuffle=True, batch_size=batch_size)

    test_loader = DataLoader(
        xy_te, shuffle=True, batch_size=batch_size)

    return train_loader, val_loader, test_loader

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

def eval(z,z_val,z_te,svmc = 0.5):
    z = list_to_np(z)
    z_val = list_to_np(z_val)
    z_te = list_to_np(z_te)
    x_train, y_train = z[:,:-1], z[:,-1]
    x_val, y_val = z_val[:, :-1], z_val[:, -1]
    x_test, y_test = z_te[:, :-1], z_te[:, -1]

    min_max_scaler = preprocessing.MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x_train)
    x_val = min_max_scaler.fit_transform(x_val)
    x_test = min_max_scaler.fit_transform(x_test)

    clf = SVC(kernel='linear', C=svmc)
    clf.fit(x_train, y_train.ravel())
    acc_tr = clf.score(x_train, y_train)

    acc_val = clf.score(x_val, y_val)
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
    return acc_tr,acc_val,acc_te,sen,spe,auc