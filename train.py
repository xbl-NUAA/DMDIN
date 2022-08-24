import torch
import torch.optim as optim
from net import DMDR
from valid import valid
from test import test
from loss_function import loss_function
from utils import divide,list_to_np,eval,loaddata




def train(model,
        train_loader, val_loader,  modalities,
        epochs, weight_coef, weight_selfExp,weight_block,  lr, mom, decay):

    optimizer = optim.SGD(model.model_list.parameters(), lr=lr, momentum=mom, weight_decay=decay)

    train_loss = []
    val_loss = []
    best_val_loss = float("inf")
    for epoch in range(epochs):
        fy_tr_list = []
        loss_epoch_train = 0
        for batch_index, (x, y) in enumerate(train_loader):
            x_list = divide(x, modalities)
            model.train()
            f_list, zf_list = model(x_list)
            loss, fx_loss = loss_function(model.self_expression.Coefficient,model.block,f_list, zf_list, weight_coef = weight_coef, weight_selfExp = weight_selfExp, weight_block = weight_block,lr = lr)
            optimizer.zero_grad()
            fx_loss.backward()
            optimizer.step()
            loss_epoch_train += loss.item()
            f = torch.cat(f_list, dim=1)
            fy_tr_list.append(torch.hstack((f,y)))
        train_loss.append(loss_epoch_train)

        loss_epoch_val, fy_val_list = valid(val_loader, model, modalities, weight_coef, weight_selfExp, weight_block, lr)
        val_loss.append(loss_epoch_val)
        if loss_epoch_val < best_val_loss:
            print("Epoch {:d}: val_loss improved from {:.4f} to {:.4f}, saving model".format(epoch + 1,
                                                                                             best_val_loss,
                                                                                             loss_epoch_val))
            best_val_loss = loss_epoch_val
            torch.save(model.state_dict(), checkpoint)
        else:
            print("Epoch {:d}: val_loss did not improve from {:.4f}".format(epoch + 1, best_val_loss))



    return train_loss,fy_tr_list, fy_val_list


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fold = 10
    path = None
    checkpoint = 'checkpoint.model'
    batch_size = 32
    channels = [90,90*2,90]
    channels_list = []
    channels_list.append(channels)
    channels_list.append(channels)
    channels_list.append(channels)
    modalities = [0,90,90*2,90*3]
    epochs = 100
    weight_coef = 1.0
    weight_selfExp = 10
    weight_block = 0.1
    show_freq = 10
    lr= 1e-6
    svmc = 0.1

    acc_tr_list = list()
    acc_te_list = list()
    auc_list = list()
    spe_list = list()
    sen_list = list()
    for i in range(fold):
        train_loader, val_loader, test_loader, class_list = loaddata(path, fold, i)
        model = DMDR(num_sample = batch_size, channels_list = channels_list,class_list = class_list)
        model.to(device)
        train_loss, fy_tr_list, fy_val_list = train(model, train_loader, val_loader,modalities, epochs, weight_coef=weight_coef, weight_selfExp=weight_selfExp,
              weight_block = weight_block, lr = lr, mom= 0.9, decay= 5e-4)

        checkpoint_ = torch.load(checkpoint)
        model.load_state_dict(checkpoint_)
        loss_epoch_test, fy_te_list = test(test_loader, model, modalities, weight_coef, weight_selfExp, weight_block, lr)
        acc_tr, acc_val, acc_te, sen, spe, auc = eval(fy_tr_list, fy_val_list, fy_te_list)
        print('acc_tr = ', acc_tr, 'acc_val', acc_val, 'acc_te =', acc_te, 'sen = ', sen, 'spe = ', spe, 'auc = ', auc)

        acc_tr_list.append(acc_tr)
        acc_te_list.append(acc_te)
        auc_list.append(auc)
        spe_list.append(spe)
        sen_list.append(sen)

    print('our_acc_tr', sum(acc_tr_list) / fold)
    print('our_acc_te', sum(acc_te_list) / fold)
    print('our_sen_tr', sum(sen_list) / fold)
    print('our_spe_tr', sum(spe_list) / fold)
    print('our_auc_tr', sum(auc_list) / fold)
