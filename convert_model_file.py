import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Conv2dSamePad(nn.Module):
    def __init__(self, kernel_size, stride):
        super(Conv2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        out_height = math.ceil(float(in_height) / float(self.stride[0]))
        out_width = math.ceil(float(in_width) / float(self.stride[1]))
        pad_along_height = ((out_height - 1) * self.stride[0] + self.kernel_size[0] - in_height)
        pad_along_width = ((out_width - 1) * self.stride[1] + self.kernel_size[1] - in_width)
        pad_top = math.floor(pad_along_height / 2)
        pad_left = math.floor(pad_along_width / 2)
        pad_bottom = pad_along_height - pad_top
        pad_right = pad_along_width - pad_left
        return F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], 'constant', 0)


class ConvTranspose2dSamePad(nn.Module):
    def __init__(self, output_size):
        super(ConvTranspose2dSamePad, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        pad_height = in_height - self.output_size[0]
        pad_width = in_width - self.output_size[1]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        return x[:, :, pad_top:in_height - pad_bottom, pad_left: in_width - pad_right]


class ConvAE(nn.Module):
    def __init__(self, channels, kernels):
        super(ConvAE, self).__init__()
        assert isinstance(channels, list) and isinstance(kernels, list)
        self.encoder = nn.Sequential()
        for i in range(1, len(channels)):
            #  Each layer will divide the size of feature map by 2
            self.encoder.add_module('pad%d' % i, Conv2dSamePad(kernels[i - 1], 2))
            self.encoder.add_module(
                'conv%d' % i,
                nn.Conv2d(channels[i - 1], channels[i], kernel_size=kernels[i - 1], stride=2)
            )
            self.encoder.add_module('relu%d' % i, nn.ReLU(True))

        self.decoder = nn.Sequential()
        channels = list(reversed(channels))
        kernels = list(reversed(kernels))
        sizes = [[12, 11], [24, 21], [48, 42]]
        for i in range(len(channels) - 1):
            # Each layer will double the size of feature map
            self.decoder.add_module(
                'deconv%d' % (i + 1),
                nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=kernels[i], stride=2)
            )
            self.decoder.add_module('padd%d' % i, ConvTranspose2dSamePad(sizes[i]))
            self.decoder.add_module('relud%d' % i, nn.ReLU(True))

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        return y


if __name__ == "__main__":
    dataset = 'yaleb'
    if dataset == 'yaleb':
        ae = ConvAE(channels=[1, 10, 20, 30], kernels=[5, 3, 3])
    state_dict = ae.state_dict()

    weights = pickle.load(open('./results/models/yaleb-cae.pkl', 'rb'), encoding='latin1')

    for k1, k2 in zip(state_dict.keys(), weights.keys()):
        print(k1, k2)
        print(state_dict[k1].size(), weights[k2].shape)
        if weights[k2].ndim > 3:
            weights[k2] = np.transpose(weights[k2], [3, 2, 0, 1])
        state_dict[k1] = torch.tensor(weights[k2], dtype=torch.float32)
        print(state_dict[k1].size())

    ae.load_state_dict(state_dict)
    torch.save(state_dict, './results/models/yaleb-cae2.pkl' % dataset)
    print('Pretrained weights are converted and saved.')
