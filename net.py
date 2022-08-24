import torch
import torch.nn as nn

class  MlpNet(nn.Module):
    def __init__(self, channels):
        super(MlpNet, self).__init__()
        assert isinstance(channels, list)
        self.encoder = nn.Sequential()
        for i in range(len(channels)-1):
            self.encoder.add_module('linear%d' % i, nn.Linear(channels[i], channels[i + 1]))
            self.encoder.add_module('relu%d' % i, nn.ReLU())

    def forward(self, x):
        x = self.encoder(x)
        return x


class SelfExpression(nn.Module):
    def __init__(self, n, weight_c):
        super(SelfExpression, self).__init__()
        self.Coefficient = weight_c * torch.randn(n, n, dtype=torch.float32)

    def forward(self, x):  # shape=[n, d]
        y = torch.matmul(self.Coefficient, x)
        return y


class DMDR(nn.Module):
    def __init__(self, channels_list, num_sample, class_list, weight_c = 1e-8):
        super(DMDR, self).__init__()
        self.n = num_sample
        self.model_list = []
        for i in range(len(channels_list)):
            self.model_list.append(MlpNet(channels_list[i]))
        self.model_list = nn.ModuleList(self.model_list)
        self.self_expression = SelfExpression(self.n, weight_c)
        mask_block = torch.ones((sum(class_list),sum(class_list)))
        index = 0
        for i in range(len(class_list)):
            mask_block[index:index + class_list[i], index:index + class_list[i]] = 0
            index += class_list[i]
        self.block = mask_block + torch.eye(sum(class_list))

    def forward(self, x_list):
        f_list = []
        zf_list = []
        for x, model in zip(x_list, self.model_list):
            out_f = model(x)
            out_zf = self.self_expression(out_f)

            f_list.append(out_f)
            zf_list.append(out_zf)

        return f_list, zf_list

