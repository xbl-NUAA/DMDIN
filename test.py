import torch
from utils import divide
from loss_function import loss_function

def test(test_loader, model, modalities, weight_coef,weight_selfExp,weight_block,lr):
    model.eval()
    fy_te_list = []
    loss_epoch_test = 0
    for batch_index, (x_te, y_te) in enumerate(test_loader):
        x_te_list = divide(x_te, modalities)
        f_te_list, zf_te_list = model(x_te_list)

        loss_test, _ = loss_function(model.self_expression.Coefficient,model.block,f_te_list, zf_te_list, weight_coef=weight_coef,
                                  weight_selfExp=weight_selfExp, weight_block=weight_block, lr=lr)
        f_te = torch.cat(f_te_list, dim=1)
        fy_te_list.append(torch.hstack((f_te, y_te)))
        loss_epoch_test += loss_test.item()

    return loss_epoch_test, fy_te_list
