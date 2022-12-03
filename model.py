import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
from torchvision.models.mobilenetv3 import mobilenet_v3_small, mobilenet_v3_large


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(
            nn.Linear(512, 512),
            nn.Hardswish(),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(512, feature_dim)
        )
        # self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
        #                        nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class Model_mobilenetv3_small(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model_mobilenetv3_small, self).__init__()

        self.f = []
        for name, module in mobilenet_v3_small().named_children():
            if name != 'classifier':
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(
                nn.Linear(576, 1024),
                nn.Hardswish(),
                nn.Dropout(0.2, inplace=True),
                nn.Linear(1024, feature_dim)
            )

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

class Model_mobilenetv3_large(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model_mobilenetv3_large, self).__init__()

        self.f = []
        for name, module in mobilenet_v3_large().named_children():
            if name != 'classifier':
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(
                nn.Linear(960, 1024),
                # nn.Linear(768, 1024),
                nn.Hardswish(),
                nn.Dropout(0.2, inplace=True),
                nn.Linear(1024, feature_dim)
            )

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)