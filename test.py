import torch
from utils import divide
from loss_function import loss_function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def test(test_loader, model, modalities, weight_coef,weight_selfExp,weight_block,lr):
    model.eval()
    f_te_list = []
    fy_te_list = []
    loss_epoch_test = 0
    for batch_index, (x_te, y_te) in enumerate(test_loader):
        x_te.to(device)
        y_te.to(device)
        x_te_list = divide(x_te, modalities)

        for i in range(len(x_te_list)):
            f_te = model.model_list[i](x_te_list[i].float().to(device))
            f_te_list.append(f_te)
        f_te = torch.cat(f_te_list, dim=1)
        fy_te_list.append(torch.hstack((f_te, y_te.to(device))))


    return fy_te_list
