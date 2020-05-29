from abc import abstractmethod

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

from torchsummary import summary

class InteractionNet(nn.Module):
    def __init__(self,
                 conv_config, #[16, 'M', 32, 'M', 64, 'M'],
                 dense_config, #[1000, 100, 100],
                 conv_kernel_size=3,
                 conv_stride=1,
                 maxpool_kernel_size=2,
                 maxpool_stride=2,
                 in_nodes=100,
                 out_nodes=2,
                 prob_drop=0.5,
                 activation='relu', # 'relu', 'elu', 'leaky_relu'
                 use_dropout=True,
                 use_batch_norm=True,
                 init_weights=True, 
                 device='cuda', 
                 verbose=False
                ):
        super(InteractionNet, self).__init__()

        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.prob_drop = prob_drop

        self.use_dropout = use_dropout

        self.in_channels = 1 # input dimension

        if activation == "relu":
            self.activation_fn = nn.ReLU
        elif activation == "elu":
            self.activation_fn = nn.eLU
        elif activation == "leaky_relu":
            self.activation_fn = nn.LeakyRelu
        else:
            self.activation_fn = nn.ReLU

        self.isfeasible = True
        self.conv_layers, h_out, out_channels = self.make_conv_layers(conv_config, use_batch_norm)
        self.dense_layers = self.make_dense_layers(dense_config, h_out, out_channels)

        self.to(torch.device(device))
        
        # self.conv1 = nn.Conv2d(1, 16, conv_kernel_size, conv_stride)
        # self.conv1_bn = nn.BatchNorm2d(16)
        # self.max1 = nn.MaxPool2d(maxpool_kernel_size, maxpool_stride)
        # self.conv2 = nn.Conv2d(16, 32, conv_kernel_size, conv_stride)
        # self.conv2_bn = nn.BatchNorm2d(32)
        # self.max2 = nn.MaxPool2d(maxpool_kernel_size, maxpool_stride)
        # self.conv3 = nn.Conv2d(32, 64, conv_kernel_size, conv_stride)
        # self.conv3_bn = nn.BatchNorm2d(64)
        # self.max3 = nn.MaxPool2d(maxpool_kernel_size, maxpool_stride)
        # self.fc1 = nn.Linear(10*10*64, fc1_nodes)
        # self.fc2 = nn.Linear(fc1_nodes, fc2_nodes)
        # self.fc3 = nn.Linear(fc2_nodes, out_nodes)
        # self.drop_out = nn.Dropout(prob_drop)
        
        self.total_params, self.total_size = summary(self, (1, 100, 100), device=device, verbose=verbose)

        if init_weights:
            self.__initialize_weights()
    
    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def __get_size(self, h, kernel_size, stride, padding=0, dilation=1):
        return int(np.floor((h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))

    def make_conv_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = self.in_channels
        h_out = self.in_nodes
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=self.maxpool_kernel_size, stride=self.maxpool_stride)]
                h_out = self.__get_size(h_out, self.maxpool_kernel_size, self.maxpool_stride)
            elif v == 'A':
                layers += [nn.AvgPool2d(kernel_size=self.maxpool_kernel_size, stride=self.maxpool_stride)]
                h_out = self.__get_size(h_out, self.maxpool_kernel_size, self.maxpool_stride, dilation=1)
            else:
                out_channels = v
                conv2d = nn.Conv2d(in_channels, out_channels, self.conv_kernel_size, stride=self.conv_stride, padding=0, dilation=1, groups=1, bias=True)
                if batch_norm:
                    # TODO order of batch norm and relu
                    # layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    layers += [conv2d, self.activation_fn(inplace=True), nn.BatchNorm2d(v)]
                else:
                    layers += [conv2d, self.activation_fn(inplace=True)]
                h_out = self.__get_size(h_out, self.conv_kernel_size, self.conv_stride)
                in_channels = v
            if h_out <= 0:
                print("[ERROR] h_out <= 0")
                self.isfeasible = False
        # layers += [nn.AdaptiveAvgPool2d(h_out, h_out)]
        return nn.Sequential(*layers), h_out, out_channels

    def make_dense_layers(self, cfg, h_in, in_channels):
        ## dense layers
        in_nodes = in_channels * h_in * h_in
        layers = []
        for _nodes in cfg:
            layers += [nn.Linear(in_nodes, _nodes), self.activation_fn(True)]
            if self.use_dropout: layers += [nn.Dropout(self.prob_drop)]
            in_nodes = _nodes
        layers += [nn.Linear(in_nodes, self.out_nodes)]
        return nn.Sequential(*layers)
        
    @abstractmethod
    def forward(self, x): raise NotImplementedError

class InteractionValueNet(InteractionNet):
    def __init__(self, **kw):
        super().__init__(**kw)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.dense_layers(x)
        return x

class InteractionModeNet2D(InteractionNet):
    def __init__(self, **kw):
        super().__init__(**kw)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.dense_layers(x)
        return F.log_softmax(x, dim=1).view(-1, self.out_nodes//2, 2)

class InteractionModeNet1D(InteractionNet):
    def __init__(self, **kw):
        super().__init__(**kw)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.dense_layers(x)
        return F.log_softmax(x, dim=1)

def model_test(arch, conv_config=[16, 'M', 32, 'M', 64, 'M'], dense_config=[1000, 100, 100]):
    net = arch(conv_config=conv_config, dense_config=dense_config)
    print(net)
    y = net(torch.randn(1,1,100,100))
    print('output size:', y.size())
