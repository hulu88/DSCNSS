'''
DSCNSS
'''


import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from post_clustering import spectral_clustering, acc, nmi, ari
from utils import ConvTranspose2dSamePad, Conv2dSamePad, SelfExpression, load_YaleB


class ConvAE(nn.Module):
    def __init__(self, channels, kernels):
        """
        :param channels: a list containing all channels including the input image channel (1 for gray, 3 for RGB)
        :param kernels:  a list containing all kernel sizes, it should satisfy: len(kernels) = len(channels) - 1.
        """
        super(ConvAE, self).__init__()
        assert isinstance(channels, list) and isinstance(kernels, list)
        self.encoder = nn.Sequential()
        for i in range(1, len(channels)):
            self.encoder.add_module('pad%d' % i, Conv2dSamePad(kernels[i - 1], 2))
            self.encoder.add_module( 'conv%d' % i, nn.Conv2d(channels[i - 1], channels[i], kernel_size=kernels[i - 1], stride=2))
            self.encoder.add_module('relu%d' % i, nn.ReLU(True))

        channels = list(reversed(channels))
        kernels = list(reversed(kernels))
        sizes = [[12, 11], [24, 21], [48, 42]]
        self.decoder = nn.Sequential()
        for i in range(len(channels) - 1):
            self.decoder.add_module('deconv%d' % (i + 1), nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=kernels[i], stride=2))
            self.decoder.add_module('padd%d' % i, ConvTranspose2dSamePad(sizes[i]))
            self.decoder.add_module('relud%d' % i, nn.ReLU(True))

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y, z


class DSCNet(nn.Module):
    def __init__(self, channels, kernels, num_sample):
        super(DSCNet, self).__init__()
        self.n = num_sample
        self.ae = ConvAE(channels, kernels)
        self.self_expression = SelfExpression(self.n)

    def encoder(self, input):
        z = self.ae.encoder(x)
        self.shape = z.shape
        z = z.view(self.n, -1)  # shape=[n, d]
        return z

    def decoder(self, input):
        z_recon_reshape = input.view(self.shape)
        x_recon = self.ae.decoder(z_recon_reshape)  # shape=[n, c, w, h]
        return x_recon

    def forward(self, x):  # shape=[n, c, w, h]
        # ??????
        z = self.encoder(x)
        # ???????????????
        z_recon = self.self_expression(z)  # shape=[n, d]
        # ??????
        x_recon = self.decoder(z_recon)
        return x_recon, z, z_recon

    def loss_fn(self, x, x_recon, z, z_recon, weight_coef, weight_selfExp):
        loss_ae = 0.5 * F.mse_loss(x_recon, x, reduction='sum')  # ????????????
        loss_coef = torch.sum(torch.pow(self.self_expression.Coefficient, 2))  # ??????C??????????????????
        loss_selfExp = 0.5 * F.mse_loss(z_recon, z, reduction='sum')  # ???????????????
        loss = loss_ae + weight_coef * loss_coef + weight_selfExp * loss_selfExp
        loss /= x.size(0)
        return loss


class DSCNSS(nn.Module):
    def __init__(self, channels, kernels, num_sample, num_class=38):
        super(DSCNSS, self).__init__()
        self.dsc = DSCNet(channels, kernels, num_sample)
        self.self_expression = self.dsc.self_expression
        self.fc1 = nn.Linear(1080, 1024)
        self.fc2 = nn.Linear(1024, num_class)

    def forward(self, x, method):
        z = self.dsc.encoder(x)
        if method == 1:  # ????????????
            # ???????????????
            z_recon = self.dsc.self_expression(z)  # shape=[n, d]
            # ??????
            x_recon = self.dsc.decoder(z_recon)
            return x_recon, z, z_recon
        else:  # ????????????
            # print(z.shape)
            feature = self.fc1(z)
            out = self.fc2(feature)
            return out

    def loss_fn(self, x, x_recon, z, z_recon, weight_coef, weight_selfExp):
        loss_ae = 0.5 * F.mse_loss(x_recon, x, reduction='sum')  # ????????????
        loss_coef = torch.sum(torch.pow(self.self_expression.Coefficient, 2))  # ??????C??????????????????
        loss_selfExp = 0.5 * F.mse_loss(z_recon, z, reduction='sum')  # ???????????????
        loss = loss_ae + weight_coef * loss_coef + weight_selfExp * loss_selfExp
        loss /= x.size(0)
        return loss

    def loss_fn_dense(self, out, y, x_p_index, x_n_index, weight_ce, weight_tr):
        self.margin = 0.5
        loss_ce = 0.5 * F.cross_entropy(out, y, reduction='sum')
        loss_tr = 0
        for i in range(len(C)):
            x_i = out[i]
            x_p = out[x_p_index[i]]
            x_n = out[x_n_index[i]]
            pos_dist = (x_i - x_p).pow(2).sum(0)/2432
            neg_dist = (x_i - x_n).pow(2).sum(0)/2432
            loss_tr += F.relu(pos_dist - neg_dist + self.margin)
        loss_tr = loss_tr / x.size(0)
        loss = weight_ce * loss_ce + weight_tr * loss_tr
        loss /= x.size(0)
        return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DSCNet')
    parser.add_argument('--dataset', default='YaleB', choices=['YaleB'])
    parser.add_argument('--epochs', default=40000, type=int)  # 1000
    parser.add_argument('--learning_rate', default=1.0e-4, type=float)  # 1.0e-3
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--show_freq', default=100, type=int)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='./results')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args)

    # ?????????????????????
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.dataset == 'YaleB':
        # load data
        x, y = load_YaleB()  # ???????????????
        print(f"x.shape={x.shape}, y.shape={y.shape}, n_class={len(np.unique(y))}")
        # print(np.unique(y))

        # network and optimization parameters
        channels = [1, 10, 20, 30]
        kernels = [5, 3, 3]
        num_class = len(set(y))
        num_sample = x.shape[0]
        args.ae_weights = "./results/models/yaleb-cae.pkl"

        # post clustering parameters
        dim_subspace = 10  # dimension of each subspace
        ro = 3.5
    else:
        print("????????????????????????")
        exit(1)

    if args.ae_weights is not None:
        if not os.path.exists(args.ae_weights):
            print("????????????????????????")
        else:
            print(f"????????????????????????{args.ae_weights}")
    else:
        print("????????????????????????????????????")

    # ???????????????
    weight_coef = 1.0
    weight_selfExp = 1.0 * 10 ** (num_class / 10.0 - 3.0)
    alpha = max(0.4 - (num_class - 1) / 10 * 0.1, 0.1)

    # ???????????????
    model = DSCNSS(num_sample=num_sample, channels=channels, kernels=kernels, num_class=num_class)
    model.to(args.device)
    print(model)
    # summary(model=model.to(args.device), input_size=(1, 48, 42), batch_size=2432)  # ??????????????????

    # ?????????????????????
    model.dsc.ae.load_state_dict(torch.load("./results/models/yaleb-cae.pkl"))  # ?????????????????????
    print("Pretrained ae weights are loaded successfully.")

    # ??????????????????
    print('='*20, 'Train on DSCNSS', '='*20)

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=args.device)
    y = y - y.min()
    if isinstance(y, torch.Tensor):
        y = y.to('cpu').numpy()
    acc_, nmi_, ari_, loss_record = 0., 0., 0., []

    for total_i in range(5):
        # args.epochs, args.show_freq = 1000, 100
        # ?????????????????????(epoch=2500)?????????????????????(epoch=10)
        args.epochs = 2500 if total_i == 0 else 10
        args.show_freq = 100 if total_i == 0 else 1

        for layer in [model.dsc, model.self_expression]:
            for name, value in layer.named_parameters():
                value.requires_grad = True
        for layer in [model.fc1, model.fc2]:
            for name, value in layer.named_parameters():
                value.requires_grad = False
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(params, lr=args.learning_rate)

        for epoch in range(1, args.epochs+1):
            x_recon, z, z_recon = model(x, 1)
            loss = model.loss_fn(x, x_recon, z, z_recon, weight_coef=weight_coef, weight_selfExp=weight_selfExp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_record.append(loss.item())

            if epoch % args.show_freq == 0 or epoch == args.epochs:
                C = model.self_expression.Coefficient.detach().to('cpu').numpy()
                y_pred = spectral_clustering(C, num_class, dim_subspace, alpha, ro)
                acc_, nmi_, ari_ = acc(y, y_pred), nmi(y, y_pred), ari(y, y_pred)
                print('Epoch %02d: loss=%.4f, acc=%.4f, clustering error=%.4f' % (epoch, loss.item(), acc_, (1-acc_)*100))
                # # ???????????????
                # if epoch % 100 == 0:
                #     torch.save(model.state_dict(), args.save_dir + '/models/yaleb-dsc.pkl')

        print('%d subjects:' % num_class)
        print('Acc: %.4f%%' % (acc_ * 100), 'NMI: %.4f%%' % (nmi_ * 100), 'ARI: %.4f%%' % (ari_ * 100), 'Clustering Error: %.4f%%' % ((1-acc_) * 100))


        args.epochs, args.show_freq = 1500, 10
        # ??????????????????
        x_p_index, x_n_index = [], []
        for index in range(len(C)):
            y_pred_temp = np.zeros(shape=(len(C)), dtype=np.int32)
            # y_C = y_pred[index]  # ???index?????????????????????(????????????????????????????????????)
            y_pred_temp[y_pred == y_pred[index]] = 1
            # ?????????????????????(???????????????????????????????????????)
            x_p = y_pred_temp * C[index]
            x_p[x_p == 0] = 1
            x_p_index.append(np.argmin(x_p))
            # ?????????????????????(???????????????????????????????????????)
            y_pred_temp = np.logical_not(y_pred_temp)
            x_n = y_pred_temp * C[index]
            x_n_index.append(np.argmax(x_n))
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.tensor(y_pred, dtype=torch.long, device=args.device)
        for layer in [model.dsc, model.self_expression]:
            for name, value in layer.named_parameters():
                value.requires_grad = False
        for layer in [model.fc1, model.fc2]:
            for name, value in layer.named_parameters():
                value.requires_grad = True
        # optimizer2 = torch.optim.Adam(model.parameters(), lr=1e-5)
        # optimizer2 = torch.optim.Adam([{'params': model.dsc.parameters()},
        #                                {'params': model.fc1.parameters(), 'lr': 1e-10},
        #                                {'params': model.fc2.parameters(), 'lr': 1e-10}], lr=1e-3)
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer2 = optim.Adam(params, lr=1e-3)
        # loss_func = torch.nn.CrossEntropyLoss()  # ???????????????
        for epoch in range(1, args.epochs + 1):
            out = model(x, 2)
            # loss = loss_func(out, y_pred)
            loss = model.loss_fn_dense(out, y_pred, x_p_index, x_n_index, weight_ce=0.7, weight_tr=0.3)
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()

            if epoch % args.show_freq == 0 or epoch == args.epochs:
                _, y_pred_dense = torch.max(out, 1)  # ????????????????????????????????????
                y_d, y_c = y_pred_dense.cpu().numpy(), y_pred.cpu().numpy()  # Tensor -> numpy ????????????????????????

                ''' ????????????
                acc_ : ?????????????????????????????????????????????????????????(?????????????????????????????????????????????)
                acc_2 ?????????????????????????????????(????????????)??????????????????????????????????????????(?????????????????????????????????????????????????????????????????????)
                '''
                acc_, nmi_, ari_, acc_2 = acc(y, y_d), nmi(y, y_d), ari(y, y_d), acc(y_c, y_d)
                print('Epoch %02d: loss=%.4f, acc_cluster_dense=%.4f, acc_oral_dense=%.4f, clustering error=%.4f' %
                      (epoch, loss.item(), acc_2, acc_, (1-acc_)*100))
                # ?????????????????????????????????????????????????????????????????????????????????????????????????????????
                if acc_2 > 0.9960:
                    epoch = epoch if acc_2 <= 0.9950 else args.epochs
                    break
                # np.savetxt("./y_pred_dense.csv", y_pred_dense, delimiter=',')

        print('%d subjects:' % num_class)
        print('Acc: %.4f%%' % (acc_ * 100), 'NMI: %.4f%%' % (nmi_ * 100), 'ARI: %.4f%%' % (ari_ * 100), 'Clustering Error: %.4f%%' % ((1-acc_) * 100))




