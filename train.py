import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from net import DMDR
from valid import valid
from test import test
from loss_function import loss_function
from utils import divide,eval,loaddata,processdata


def train(model,
        train_loader, val_loader,  modalities,
        epochs, weight_coef, weight_selfExp,weight_block,  lr, mom, decay,svmc):

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=decay)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_acc_val = 0
    for epoch in range(1, epochs + 1):
        fy_tr_list = []
        loss_epoch_train = 0
        for batch_index, (x, y) in enumerate(train_loader):
            x.to(device)
            y.to(device)
            model.train()
            x_list = divide(x, modalities)
            f_list, zf_list = model(x_list)
            loss = loss_function(model.self_expression.Coefficient,model.block,f_list, zf_list,weight_coef, weight_selfExp, weight_block)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch_train += loss.item()
            f = torch.cat(f_list, dim=1)
            fy_tr_list.append(torch.hstack((f,y.to(device))))

        if val_loader != None:
            fy_val_list = valid(val_loader, model, modalities, weight_coef, weight_selfExp, weight_block)
            acc_tr, acc_val, _, _, _ = eval(fy_tr_list, fy_val_list, svmc)
            if acc_val > max_acc_val:
                max_acc_val = acc_val
                torch.save(model.state_dict(), checkpoint)


    return fy_tr_list,acc_val


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fold = 10
    path ='data.npy'
    checkpoint = 'checkpoint.model'

    out = 40
    channels1 = [3, 128 * 2, out]
    channels2 = [93, 128 * 2,  out]
    channels3 = [93, 128 * 2, out]
    channels_list = []
    channels_list.append(channels1)
    channels_list.append(channels2)
    channels_list.append(channels3)
    modalities = [0,3, 3 + 93,93 * 2 + 3]
    epochs = 110
    weight_coef = 1
    weight_selfExp = 0.1
    weight_block = 1
    show_freq = 1
    lr= 1e-5
    svmc = 0.3
    mom_list = 0.9
    decay_list = 5e-4

    acc_tr_list = list()
    acc_te_list = list()
    auc_list = list()
    spe_list = list()
    sen_list = list()

    fold_acc_tr_list = np.empty((0, epochs))
    fold_acc_te_list = np.empty((0, epochs))
    fold_sen_list = np.empty((0, epochs))
    fold_spe_list = np.empty((0, epochs))
    fold_auc_list = np.empty((0, epochs))

    x_train_fold, y_train_fold, x_val_fold, y_val_fold, x_test_fold, y_test_fold = processdata(path, fold)

    for i in range(fold):
        seed = 66
        print('seed is {}'.format(seed))
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        train_loader, val_loader, test_loader, class_list = loaddata(x_train_fold, y_train_fold, x_val_fold, y_val_fold,
                                                                     x_test_fold, y_test_fold, i)
        model = DMDR(num_sample=class_list[0] + class_list[1], channels_list=channels_list, class_list=class_list)
        model.to(device)
        fy_tr_list, acc_val = train(model, train_loader, val_loader, modalities,epochs,weight_coef=weight_coef,
                                                                                                 weight_selfExp=weight_selfExp,
                                                                                                 weight_block=weight_block,
                                                                                                 lr=lr, mom=mom_list, decay=decay_list,
                                                                                                 svmc=0.3)

        checkpoint_ = torch.load(checkpoint)
        model.load_state_dict(checkpoint_)
        fy_te_list = test(test_loader, model, modalities, weight_coef, weight_selfExp, weight_block, lr)
        acc_tr, acc_te, sen, spe, auc = eval(fy_tr_list, fy_te_list, svmc)

        print('fold = ', i, 'acc_tr = ', acc_tr,'acc_val = ', acc_val,  'acc_te =', acc_te, 'sen = ', sen, 'spe = ', spe, 'auc = ', auc)

        acc_tr_list.append(acc_tr)
        acc_te_list.append(acc_te)
        auc_list.append(auc)
        spe_list.append(spe)
        sen_list.append(sen)

    print('acc_tr', sum(acc_tr_list) / fold)
    print('acc_te', sum(acc_te_list) / fold)
    print('sen_tr', sum(sen_list) / fold)
    print('spe_tr', sum(spe_list) / fold)
    print('auc_tr', sum(auc_list) / fold)


