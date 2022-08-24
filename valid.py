import torch
from utils import divide
from loss_function import loss_function

def valid(val_loader, model, modalities, weight_coef,weight_selfExp,weight_block,lr):
    fy_val_list = []
    loss_all_val = 0
    with torch.no_grad():
        for batch_index, (x_val, y_val) in enumerate(val_loader):
            model.eval()
            x_val_list = divide(x_val, modalities)
            f_val_list, zf_val_list = model(x_val_list)
            loss_val, _ = loss_function(model.self_expression.Coefficient,model.block,f_val_list, zf_val_list, weight_coef=weight_coef,
                                         weight_selfExp=weight_selfExp, weight_block=weight_block, lr=lr)

            loss_all_val += loss_val.item()
            f_val = torch.cat(f_val_list, dim=1)
            fy_val_list.append(torch.hstack((f_val, y_val)))

    return loss_all_val, fy_val_list
