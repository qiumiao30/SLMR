# -*- coding: utf-8 -*-
# @Time    : 2021/10/14 14:54
# @File    : se_in_gru_zhiqian.py
import torch.nn as nn
import torch
import numpy as np
device = "cuda"


def pooling():
    return nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

# def conv1x1(in_planes, out_planes, stride=1):
#     return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
#
# def conv3x3(in_planes, out_planes, stride=1, groups=1):
#     return nn.Conv1d(in_planes, out_planes, kernel_size=3, dilation=2, stride=stride, padding=2, groups=groups, bias=False)


# class Res2NetBottleneck(nn.Module):
#     def __init__(self, inplanes, bottleneck_planes, outplanes):
#         super(Res2NetBottleneck, self).__init__()

#         self.conv1 = conv1x1(inplanes, bottleneck_planes)
#         self.conv2 = conv3x3(bottleneck_planes, bottleneck_planes)
#         self.conv3 = conv5x5(bottleneck_planes, outplanes)

#         self.deconv1 = deconv5x5(outplanes, bottleneck_planes)
#         self.deconv2 = deconv3x3(bottleneck_planes, bottleneck_planes)
#         self.deconv3 = deconv1x1(bottleneck_planes, inplanes)

#     def forward(self, x):

#         x = x.permute(0, 2, 1)

#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = self.conv3(out)

#         out = self.deconv1(out)
#         out = self.deconv2(out)
#         out = self.deconv3(out)

#         return out.permute(0,2,1)

#
# def main():
#     model = Res2NetBottleneck(inplanes=51, bottleneck_planes=60, outplanes=32)
#     print(model)
#
#     input = torch.randn(128, 64, 51)
#     # input = torch.randn(128, 51, 64)
#     y = model(input)
#     print(y.shape)
#
#
# if __name__ == "__main__":
#     main()

def split(seq_size):
    idx_odd = torch.Tensor(np.arange(0, seq_size, 2)).to(device)  # because 1th, 3th... element ist 0th, 2th ... in python
    idx_odd = idx_odd.long()

    # indexing for F_odd
    idx_even = torch.Tensor(np.arange(1, seq_size, 2)).to(device)
    idx_even = idx_even.long()

    # create F_even and F_odd
    return idx_even.to(device), idx_odd.to(device)

def pooling():
    return nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, kernel_size=3, padding=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, dilation=1, stride=stride, padding=padding, groups=groups, bias=False)

class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        #x = torch.softmax(x, dim=1)

        score = x.detach().cpu().numpy()
        np.save("score", score)
        # attention = torch.softmax(x, dim=1)
        # print(attention)
        #
        # print(x[0,:,:])
        #
        # print(attention[0,:,:].sum())

        return input * x

class Forecasting_Model(nn.Module):
    """Forecasting model (fully-connected network)
    :param in_dim: number of input features
    :param hid_dim: hidden size of the FC network
    :param out_dim: number of output features
    :param n_layers: number of FC layers
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(Forecasting_Model, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))

        layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)
        return self.layers[-1](x)

class RNNDecoder(nn.Module):
    """GRU-based Decoder network that converts latent vector into output
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(RNNDecoder, self).__init__()
        self.in_dim = in_dim
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.rnn = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        decoder_out, _ = self.rnn(x)
        return decoder_out

class ReconstructionModel(nn.Module):
    """Reconstruction Model
    :param window_size: length of the input sequence
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param in_dim: number of output features
    :param dropout: dropout rate
    """

    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ReconstructionModel, self).__init__()
        self.window_size = window_size
        self.decoder = RNNDecoder(in_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        # x will be last hidden state of the GRU layer
        h_end = x
        h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1)

        decoder_out = self.decoder(h_end_rep)
        out = self.fc(decoder_out)
        return out

class Res2NetBottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 out_dim,
                 window,
                 gru_hid_dim,
                 gru_n_layers,
                 fc_hid_dim,
                 fc_n_layers,
                 recon_hid_dim,
                 recon_n_layers,
                 downsample,
                 dropout,
                 stride=1,
                 scales=4,
                 groups=1,
                 se=False,
                 norm_layer=None):
        super(Res2NetBottleneck, self).__init__()
        if planes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        bottleneck_planes = groups * planes
        self.pool = pooling()
        self.window=window
        self.pool = pooling()
        self.conv1 = conv1x1(inplanes, bottleneck_planes, stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = nn.ModuleList(
            [conv3x3(bottleneck_planes // scales, bottleneck_planes // scales, groups=groups,
                     kernel_size=i * 2 + 1, padding=i) for i in range(scales - 1)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales - 1)])
        self.conv3 = conv1x1(bottleneck_planes, inplanes)
        self.bn3 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEModule(inplanes) if se else None
        self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * self.expansion, stride),
                norm_layer(planes * self.expansion),
            )
        self.stride = stride
        self.scales = scales
        self.idx_even, self.idx_odd = split(window)
        self.gru = nn.GRU(inplanes, gru_hid_dim, gru_n_layers, batch_first=True)
        self.forecasting_model = Forecasting_Model(in_dim=gru_hid_dim, hid_dim=fc_hid_dim, out_dim=out_dim, n_layers=fc_n_layers, dropout=dropout)
        self.recon_model = ReconstructionModel(window, in_dim=gru_hid_dim, hid_dim=recon_hid_dim, out_dim=out_dim, n_layers=recon_n_layers, dropout=dropout)

    def forward(self, x):
        identity = x.permute(0, 2, 1)
        x = x.permute(0, 2, 1)

        identity_even = identity[:, :, self.idx_even]
        identity_odd = identity[:, :, self.idx_odd]

        F_even = x[:, :, self.idx_even]
        F_odd = x[:, :, self.idx_odd]

        out_even = self.conv1(F_even)
        out_even = self.pool(out_even)
        out_even = self.bn1(out_even)
        out_even = self.relu(out_even)

        xs_even = torch.chunk(out_even, self.scales, 1)
        ys_even = []
        for s in range(self.scales):
            if s == 0:
                ys_even.append(xs_even[s])
            elif s == 1:
                ys_even.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs_even[s]))))
            else:
                ys_even.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs_even[s] + ys_even[-1]))))
        out_even = out_even - torch.cat(ys_even, 1)

        out_even = self.conv3(out_even)
        # print(out.shape)

        # if self.se is not None:
        #     out_even = self.se(out_even)

        # if self.downsample is not None:
        #     identity_even = self.downsample(identity_even)
        # print(identity.shape)

        out_odd = self.conv1(F_odd)
        out_odd = self.pool(out_odd)
        out_odd = self.bn1(out_odd)
        out_odd = self.relu(out_odd)

        xs_odd = torch.chunk(out_odd, self.scales, 1)
        ys_odd = []
        for s in range(self.scales):
            if s == 0:
                ys_odd.append(xs_odd[s])
            elif s == 1:
                ys_odd.append(self.relu(self.bn2[s - 1](self.pool(self.conv2[s - 1](xs_odd[s])))))
            else:
                ys_odd.append(self.relu(self.bn2[s - 1](self.pool(self.conv2[s - 1](xs_odd[s] + ys_odd[-1])))))
        out_odd = out_odd - torch.cat(ys_odd, 1)

        out_odd = self.conv3(out_odd)
        # print(out.shape)

        # if self.se is not None:
        #     out_odd = self.se(out_odd)

        # if self.downsample is not None:
        #     identity_odd = self.downsample(identity_odd)
        # print(identity.shape)

        out_even += identity_odd
        out_even = self.pool(out_even)
        out_even = self.bn3(out_even)
        out_even = self.relu(out_even)
        out_even = out_even.permute(0, 2, 1)

        out_odd += identity_even
        out_odd = self.pool(out_odd)
        out_odd = self.bn3(out_odd)
        out_odd = self.relu(out_odd)
        out_odd = out_odd.permute(0, 2, 1)

        # for i, v in enumerate(out_odd):
        #     out_even.insert(2*i+1, v)

        out = torch.randn(x.shape[0], self.window, out_odd.shape[2]).to(device)
        n = self.window
        out[:, 0:n:2, :] = out_odd
        out[:, 1:n:2, :] = out_even
        out = out.permute(0,2,1)

        if self.se is not None:
            out = self.se(out)
        out = out.permute(0, 2, 1)
        #
        out, h_end = self.gru(out)
        h_end = h_end.view(x.shape[0], -1)  # Hidden state for last timestamp
        predictions = self.forecasting_model(h_end)  # [batch, features], and DATA:MSL/SMAP-->[batch, 1]
        recons = self.recon_model(h_end)

        return predictions, recons
