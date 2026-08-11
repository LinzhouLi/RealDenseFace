import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class Inception(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1x1 = BasicConv2d(128, 32, kernel_size=1, padding=0)
        self.branch1x1_2 = BasicConv2d(128, 32, kernel_size=1, padding=0)
        self.branch3x3_reduce = BasicConv2d(128, 24, kernel_size=1, padding=0)
        self.branch3x3 = BasicConv2d(24, 32, kernel_size=3, padding=1)
        self.branch3x3_reduce_2 = BasicConv2d(128, 24, kernel_size=1, padding=0)
        self.branch3x3_2 = BasicConv2d(24, 32, kernel_size=3, padding=1)
        self.branch3x3_3 = BasicConv2d(32, 32, kernel_size=3, padding=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch1x1_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch1x1_2 = self.branch1x1_2(branch1x1_pool)
        branch3x3_reduce = self.branch3x3_reduce(x)
        branch3x3 = self.branch3x3(branch3x3_reduce)
        branch3x3_reduce_2 = self.branch3x3_reduce_2(x)
        branch3x3_2 = self.branch3x3_2(branch3x3_reduce_2)
        branch3x3_3 = self.branch3x3_3(branch3x3_2)
        return torch.cat([branch1x1, branch1x1_2, branch3x3, branch3x3_3], 1)


class FaceBoxes(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = BasicConv2d(3, 8, kernel_size=3, stride=2, padding=1)
        self.conv2 = BasicConv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = BasicConv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = BasicConv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv5 = BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.inception1 = Inception()
        self.inception2 = Inception()
        self.inception3 = Inception()
        self.conv6_1 = BasicConv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.conv6_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv7_1 = BasicConv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.conv7_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.loc, self.conf = self.multibox(num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def multibox(self, num_classes: int):
        loc_layers = [
            nn.Conv2d(128, 21 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1),
        ]
        conf_layers = [
            nn.Conv2d(128, 21 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1),
        ]
        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)

    def forward(self, x):
        sources = []
        loc = []
        conf = []
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        sources.append(x)
        x = self.conv6_1(x)
        x = self.conv6_2(x)
        sources.append(x)
        x = self.conv7_1(x)
        x = self.conv7_2(x)
        sources.append(x)

        for feat, l, c in zip(sources, self.loc, self.conf):
            loc.append(l(feat).permute(0, 2, 3, 1))
            conf.append(c(feat).permute(0, 2, 3, 1))

        loc = torch.cat([o.reshape(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.reshape(o.size(0), -1) for o in conf], 1)
        return loc.view(loc.size(0), -1, 4), self.softmax(conf.view(conf.size(0), -1, self.num_classes))
