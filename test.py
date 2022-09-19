# -*- coding: utf-8 -*-
# @Time    : 2021/9/13 19:05
# @File    : test.py
import torch.nn as nn
import torch

# def conv1x1(in_planes, out_planes, stride=1):
#     return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
# def conv3x3(in_planes, out_planes, stride=1, groups=1):
#     return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
# def conv5x5(in_planes, out_planes, stride=1, groups=1):
#     return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, groups=groups, bias=False)
# def conv7x7(in_planes, out_planes, stride=1, groups=1):
#     return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3, groups=groups, bias=False)
#
# from math import sqrt
#
# import torch
# import torch.nn as nn
#
# class SelfAttention(nn.Module):
#     dim_in: int
#     dim_k: int
#     dim_v: int
#
#     def __init__(self, dim_in, dim_k, dim_v):
#         super(SelfAttention, self).__init__()
#         self.dim_in = dim_in
#         self.dim_k = dim_k
#         self.dim_v = dim_v
#         self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
#         self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
#         self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
#         self._norm_fact = 1 / sqrt(dim_k)
#
#     def forward(self, x):
#         # x: batch, n, dim_in
#         batch, n, dim_in = x.shape
#         assert dim_in == self.dim_in
#
#         q = self.linear_q(x)  # batch, n, dim_k
#         k = self.linear_k(x)  # batch, n, dim_k
#         v = self.linear_v(x)  # batch, n, dim_v
#
#         dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
#         dist = torch.softmax(dist, dim=-1)  # batch, n, n
#
#         att = torch.bmm(dist, v)
#         return att
#
# class SEModule(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super(SEModule, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1, padding=0)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Conv1d(channels // reduction, channels=channels, kernel_size=1, padding=0)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, input):
#         x = self.avg_pool(input)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.sigmoid(x)
#         return input * x
#
#
# class Forecasting_Model(nn.Module):
#     """Forecasting model (fully-connected network)
#     :param in_dim: number of input features
#     :param hid_dim: hidden size of the FC network
#     :param out_dim: number of output features
#     :param n_layers: number of FC layers
#     :param dropout: dropout rate
#     """
#
#     def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
#         super(Forecasting_Model, self).__init__()
#         layers = [nn.Linear(in_dim, hid_dim)]
#         for _ in range(n_layers - 1):
#             layers.append(nn.Linear(hid_dim, hid_dim))
#
#         layers.append(nn.Linear(hid_dim, out_dim))
#
#         self.layers = nn.ModuleList(layers)
#         self.dropout = nn.Dropout(dropout)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         for i in range(len(self.layers) - 1):
#             x = self.relu(self.layers[i](x))
#             x = self.dropout(x)
#         return self.layers[-1](x)
#
# class RNNDecoder(nn.Module):
#     """GRU-based Decoder network that converts latent vector into output
#     :param in_dim: number of input features
#     :param n_layers: number of layers in RNN
#     :param hid_dim: hidden size of the RNN
#     :param dropout: dropout rate
#     """
#
#     def __init__(self, in_dim, hid_dim, n_layers, dropout):
#         super(RNNDecoder, self).__init__()
#         self.in_dim = in_dim
#         self.dropout = 0.0 if n_layers == 1 else dropout
#         self.rnn = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, dropout=self.dropout)
#
#     def forward(self, x):
#         decoder_out, _ = self.rnn(x)
#         return decoder_out
#
#
# class ReconstructionModel(nn.Module):
#     """Reconstruction Model
#     :param window_size: length of the input sequence
#     :param in_dim: number of input features
#     :param n_layers: number of layers in RNN
#     :param hid_dim: hidden size of the RNN
#     :param in_dim: number of output features
#     :param dropout: dropout rate
#     """
#
#     def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
#         super(ReconstructionModel, self).__init__()
#         self.window_size = window_size
#         self.decoder = RNNDecoder(in_dim, hid_dim, n_layers, dropout)
#         self.fc = nn.Linear(hid_dim, out_dim)
#
#     def forward(self, x):
#         # x will be last hidden state of the GRU layer
#         h_end = x
#         h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1)
#
#         decoder_out = self.decoder(h_end_rep)
#         out = self.fc(decoder_out)
#         return out
#
# class Res2NetBottleneck(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, downsample=None, stride=1, scales=4, groups=1, se=False, norm_layer=None):
#         super(Res2NetBottleneck, self).__init__()
#         if planes % scales != 0:
#             raise ValueError('Planes must be divisible by scales')
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm1d
#         bottleneck_planes = groups * planes
#         self.conv1 = conv1x1(inplanes, bottleneck_planes, stride)
#         self.bn1 = norm_layer(bottleneck_planes)
#
#
#         self.conv2 = nn.ModuleList(
#             [conv3x3(bottleneck_planes // scales, bottleneck_planes // scales, groups=groups) for _ in
#              range(scales - 1)])
#         self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales - 1)])
#
#
#         self.conv3 = nn.ModuleList(
#             [conv5x5(bottleneck_planes // scales, bottleneck_planes // scales, groups=groups) for _ in
#              range(scales - 1)])
#         self.bn3 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales - 1)])
#
#
#         self.conv4 = nn.ModuleList(
#             [conv7x7(bottleneck_planes // scales, bottleneck_planes // scales, groups=groups) for _ in
#              range(scales - 1)])
#         self.bn4 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales - 1)])
#
#
#         self.conv5 = conv1x1(bottleneck_planes, inplanes * self.expansion)
#         self.bn5 = norm_layer(inplanes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.se = SEModule(planes * self.expansion) if se else None
#         self.downsample = downsample
#         self.stride = stride
#         self.scales = scales
#         self.gru = nn.GRU(51*3, 64, batch_first=True)
#         self.forecasting_model = Forecasting_Model(in_dim=64, hid_dim=64, out_dim=51, n_layers=1, dropout=0.2)
#         self.recon_model = ReconstructionModel(64, in_dim=64, hid_dim=150, out_dim=51, n_layers=1,
#                                                dropout=0.3)
#
#     def forward(self, x):
#         identity = x.permute(0, 2, 1)
#         x = x.permute(0, 2, 1)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         xs = torch.chunk(out, self.scales, 1)
#
#
#         ys1 = []
#         ys2 = []
#         ys3 = []
#         out_all = []
#
# ####################     1
#         for s in range(self.scales):
#             if s == 0:
#                 ys1.append(xs[s])
#             elif s == 1:
#                 ys1.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s]))))
#             else:
#                 ys1.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s] + ys1[-1]))))
#         out1 = torch.cat(ys1, 1)
#
#         out1 = self.conv5(out1)
#         out1 = self.bn5(out1)
#         # print(out1.shape)
#
#         if self.se is not None:
#             out1 = self.se(out1)
#
#         if self.downsample is not None:
#             identity = self.downsample(identity)
#         # print(identity.shape)
#
#         out1 += identity
#         out1 = self.relu(out1)
#         out_all.append(out1)
#
# ######################    2
#         for s in range(self.scales):
#             if s == 0:
#                 ys2.append(xs[s])
#             elif s == 1:
#                 ys2.append(self.relu(self.bn3[s - 1](self.conv3[s - 1](xs[s]))))
#             else:
#                 ys2.append(self.relu(self.bn3[s - 1](self.conv3[s - 1](xs[s] + ys2[-1]))))
#         out2 = torch.cat(ys2, 1)
#
#         out2 = self.conv5(out2)
#         out2 = self.bn5(out2)
#         # print(out2.shape)
#
#         if self.se is not None:
#             out2 = self.se(out2)
#
#         if self.downsample is not None:
#             identity = self.downsample(identity)
#         # print(identity.shape)
#
#         out2 += identity
#         out2 = self.relu(out2)
#         out_all.append(out2)
#
#
# ########################  3
#         for s in range(self.scales):
#             if s == 0:
#                 ys3.append(xs[s])
#             elif s == 1:
#                 ys3.append(self.relu(self.bn4[s - 1](self.conv4[s - 1](xs[s]))))
#             else:
#                 ys3.append(self.relu(self.bn4[s - 1](self.conv4[s - 1](xs[s] + ys3[-1]))))
#         out3 = torch.cat(ys3, 1)
#
#         out3 = self.conv5(out3)
#         out3 = self.bn5(out3)
#         # print(out3.shape)
#
#         if self.se is not None:
#             out3 = self.se(out3)
#
#         if self.downsample is not None:
#             identity = self.downsample(identity)
#         # print(identity.shape)
#
#         out3 += identity
#         out3 = self.relu(out3)
#         # out3 = out3.permute(0, 2, 1)
#         out_all.append(out3)
#
#         out = torch.cat(out_all, 1)
#         # print("out:", out.shape)
#         out = out.permute(0, 2, 1)
#         print("out_all:", out.shape)
#
#         out, h_end = self.gru(out)
#         h_end = h_end.view(x.shape[0], -1)
#         # out = out.view(x.shape[0], -1)
#         predictions = self.forecasting_model(h_end)
#         recons = self.recon_model(h_end)
#
#         return predictions, recons

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
def conv3x3(in_planes, out_planes, stride=1, groups=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)

class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
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
    expansion = 1

    def __init__(self, inplanes, planes, downsample=None, stride=1, scales=4, groups=1, se=False, norm_layer=None):
        super(Res2NetBottleneck, self).__init__()
        if planes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        bottleneck_planes = groups * planes
        self.conv1 = conv1x1(inplanes, bottleneck_planes, stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = nn.ModuleList(
            [conv3x3(bottleneck_planes // scales, bottleneck_planes // scales, groups=groups) for _ in
             range(scales - 1)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales - 1)])
        self.conv3 = conv1x1(bottleneck_planes, inplanes * self.expansion)
        self.bn3 = norm_layer(inplanes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEModule(planes * self.expansion) if se else None
        self.downsample = downsample
        self.stride = stride
        self.scales = scales
        self.gru = nn.GRU(51, 150, batch_first=True)
        self.forecasting_model = Forecasting_Model(in_dim=150, hid_dim=150, out_dim=51, n_layers=1, dropout=0.2)
        self.recon_model = ReconstructionModel(5, in_dim=150, hid_dim=150, out_dim=51, n_layers=1, dropout=0.3)

    def forward(self, x):
        identity = x.permute(0, 2, 1)
        x = x.permute(0, 2, 1)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        xs = torch.chunk(out, self.scales, 1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)

        out = self.conv3(out)
        out = self.bn3(out)
        # print(out.shape)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(identity)
        # print(identity.shape)

        out += identity
        out = self.relu(out)
        out = out.permute(0, 2, 1)
        out, h_end = self.gru(out)
        h_end = h_end.view(x.shape[0], -1)  # Hidden state for last timestamp
        predictions = self.forecasting_model(h_end)  # [batch, features], and DATA:MSL/SMAP-->[batch, 1]
        recons = self.recon_model(h_end)

        return predictions, recons

def main():
    model = Res2NetBottleneck(inplanes=51, planes=51*6, downsample=None, stride=1, scales=3, groups=6, se=False,
                              norm_layer=None)
    print(model)
    input = torch.randn(128, 64, 51)
    y, y2 = model(input)
    print(y.shape)
    print(y2.shape)


if __name__ == "__main__":
    main()