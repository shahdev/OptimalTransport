'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg_ = {
    'VGG9': [32, 64, 'M', 128, 128, 'M', 256, 256, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}


class VGG9(nn.Module):
    def __init__(self, vgg_name='VGG9'):
        super(VGG9, self).__init__()
        self.net = self._make_layers(cfg_[vgg_name])
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(in_features=(256 * 4 * 4), out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=512, out_features=10),
        )
        #self.init_bias()  # initialize bias

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg_):
        layers = []
        in_channels = 3
        for x in cfg_:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU()]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
