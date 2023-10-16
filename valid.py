import torch
from utils import divide
from loss_function import loss_function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def valid(val_loader, model, modalities, weight_coef,weight_selfExp,weight_block):
    f_val_list = []
    fy_val_list = []
    with torch.no_grad():
        for batch_index, (x_val, y_val) in enumerate(val_loader):
            model.eval()
            x_val.to(device)
            y_val.to(device)
            x_val_list = divide(x_val, modalities)

            for i in range(len(x_val_list)):
                f_val = model.model_list[i](x_val_list[i].float().to(device))
                f_val_list.append(f_val)
            f_val = torch.cat(f_val_list, dim=1)
            fy_val_list.append(torch.hstack((f_val, y_val.to(device))))

    return fy_val_list
