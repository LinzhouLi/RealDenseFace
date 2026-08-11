import torch
from torch import nn
import torch.nn.functional as F


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = nn.ReLU(False)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(
        self, 
        features, 
        bn=False,  
        up_sample=True,
        align_corners=True,
        fuse=True
    ):
        """Init.
        
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.fuse = fuse
        self.up_sample = up_sample
        self.align_corners = align_corners

        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.resConfUnit2 = ResidualConvUnit(features, bn)
        
        if self.fuse:
            self.resConfUnit1 = ResidualConvUnit(features, bn)
            self.skip_add = nn.quantized.FloatFunctional()


    def forward(self, x1, x2=None):
        if x2 is not None:
            assert self.fuse
            res = self.resConfUnit1(x2)
            x1 = self.skip_add.add(x1, res)

        output = self.resConfUnit2(x1)
        if self.up_sample:
            output = F.interpolate(output, scale_factor=2.0, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output
    

class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        features=256, 
        use_bn=False, 
        mid_channels=[256, 512, 1024, 1024]
    ):
        super(DPTHead, self).__init__()
        
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels, out_channel, kernel_size=1, stride=1, padding=0)
            for out_channel in mid_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(mid_channels[0], mid_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(mid_channels[1], mid_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(mid_channels[3], mid_channels[3], kernel_size=3, stride=2, padding=1)
        ])

        self.layer1_rn = nn.Conv2d(mid_channels[0], features, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.layer2_rn = nn.Conv2d(mid_channels[1], features, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.layer3_rn = nn.Conv2d(mid_channels[2], features, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.layer4_rn = nn.Conv2d(mid_channels[3], features, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        
        self.refinenet1 = FeatureFusionBlock(features, use_bn)
        self.refinenet2 = FeatureFusionBlock(features, use_bn)
        self.refinenet3 = FeatureFusionBlock(features, use_bn)
        self.refinenet4 = FeatureFusionBlock(features, use_bn, fuse=False)
        
        self.output_conv1 = nn.Conv2d(features, out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, out_features):
        out = []
        for i, x in enumerate(out_features):
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.layer1_rn(layer_1)
        layer_2_rn = self.layer2_rn(layer_2)
        layer_3_rn = self.layer3_rn(layer_3)
        layer_4_rn = self.layer4_rn(layer_4)
        
        path_4 = self.refinenet4(layer_4_rn)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        path_1 = self.refinenet1(path_2, layer_1_rn)
        
        out = self.output_conv1(path_1)
        out = F.interpolate(out, scale_factor=2.0, mode="bilinear", align_corners=True)
        # out = self.output_conv2(out)
        return out
    

class FlexDPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        features=128, 
        mid_channels=[48, 96, 192, 384],
        align_corners=True,
        use_bn=False,
        dpt_mode='a'
    ):
        super(FlexDPTHead, self).__init__()
        self.align_corners = align_corners
        n_blks = len(mid_channels)

        self.dpt_mode = dpt_mode
        if dpt_mode == 'a':
            self.projects = nn.ModuleList([
                nn.Conv2d(in_channels, mid_channel, kernel_size=1, stride=1, padding=0)
                for mid_channel in mid_channels
            ])
            self.resize_layers = nn.ModuleList()
            for i in range(n_blks):
                if i == n_blks - 4: layer = nn.ConvTranspose2d(mid_channels[i], mid_channels[i], kernel_size=4, stride=4, padding=0)
                elif i == n_blks - 3: layer = nn.ConvTranspose2d(mid_channels[i], mid_channels[i], kernel_size=2, stride=2, padding=0)
                elif i == n_blks - 2: layer = nn.Identity()
                elif i == n_blks - 1: layer = nn.Conv2d(mid_channels[i], mid_channels[i], kernel_size=3, stride=2, padding=1)
                else: layer = nn.ConvTranspose2d(mid_channels[i], mid_channels[i], kernel_size=8, stride=8, padding=0)
                self.resize_layers.append(layer)
        elif dpt_mode == 'b':
            self.resize_layers = nn.ModuleList()
            for i in range(n_blks):
                if i == n_blks - 4: layer = nn.ConvTranspose2d(in_channels, mid_channels[i], kernel_size=4, stride=4, padding=0)
                elif i == n_blks - 3: layer = nn.ConvTranspose2d(in_channels, mid_channels[i], kernel_size=2, stride=2, padding=0)
                elif i == n_blks - 2: layer = nn.Conv2d(in_channels, mid_channels[i], kernel_size=1, stride=1, padding=0)
                elif i == n_blks - 1: layer = nn.Conv2d(in_channels, mid_channels[i], kernel_size=3, stride=2, padding=1)
                else: layer = nn.ConvTranspose2d(in_channels, mid_channels[i], kernel_size=8, stride=8, padding=0)
                self.resize_layers.append(layer)

        self.rn_layers = nn.ModuleList([
            nn.Conv2d(mid_channels[i], features, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
            for i in range(n_blks)
        ])

        self.refine_layers = nn.ModuleList([
            FeatureFusionBlock(features, use_bn, up_sample=(i < 4), align_corners=align_corners, fuse=(i!=0))
            for i in range(n_blks)
        ])

        self.output_conv = nn.Conv2d(features, out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, out_features):
        out = []
        for i, x in enumerate(out_features):
            if self.dpt_mode == 'a':
                x = self.projects[i](x)
            x = self.resize_layers[i](x)
            x = self.rn_layers[i](x)
            out.append(x)

        path = self.refine_layers[0](out[-1])
        for i in range(1, len(self.refine_layers)):
            path = self.refine_layers[i](path, out[-(i+1)])
        
        out = self.output_conv(path)
        out = F.interpolate(out, scale_factor=2.0, mode="bilinear", align_corners=self.align_corners)
        return out
    

class DualFlexDPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        features=128, 
        mid_channels=[48, 96, 192, 384],
        align_corners=True,
        use_bn=False,
        dpt_mode='a'
    ):
        super(DualFlexDPTHead, self).__init__()
        self.align_corners = align_corners
        n_blks = len(mid_channels)

        self.dpt_mode = dpt_mode
        if dpt_mode == 'a':
            self.projects = nn.ModuleList([
                nn.Conv2d(in_channels, mid_channel, kernel_size=1, stride=1, padding=0)
                for mid_channel in mid_channels
            ])
            self.resize_layers = nn.ModuleList()
            for i in range(n_blks):
                if i == n_blks - 4: layer = nn.ConvTranspose2d(mid_channels[i], mid_channels[i], kernel_size=4, stride=4, padding=0)
                elif i == n_blks - 3: layer = nn.ConvTranspose2d(mid_channels[i], mid_channels[i], kernel_size=2, stride=2, padding=0)
                elif i == n_blks - 2: layer = nn.Identity()
                elif i == n_blks - 1: layer = nn.Conv2d(mid_channels[i], mid_channels[i], kernel_size=3, stride=2, padding=1)
                else: layer = nn.ConvTranspose2d(mid_channels[i], mid_channels[i], kernel_size=8, stride=8, padding=0)
                self.resize_layers.append(layer)
        elif dpt_mode == 'b':
            self.resize_layers = nn.ModuleList()
            for i in range(n_blks):
                if i == n_blks - 4: layer = nn.ConvTranspose2d(in_channels, mid_channels[i], kernel_size=4, stride=4, padding=0)
                elif i == n_blks - 3: layer = nn.ConvTranspose2d(in_channels, mid_channels[i], kernel_size=2, stride=2, padding=0)
                elif i == n_blks - 2: layer = nn.Conv2d(in_channels, mid_channels[i], kernel_size=1, stride=1, padding=0)
                elif i == n_blks - 1: layer = nn.Conv2d(in_channels, mid_channels[i], kernel_size=3, stride=2, padding=1)
                else: layer = nn.ConvTranspose2d(in_channels, mid_channels[i], kernel_size=8, stride=8, padding=0)
                self.resize_layers.append(layer)

        self.rn_layers = nn.ModuleList([
            nn.Conv2d(mid_channels[i], features, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
            for i in range(n_blks)
        ])

        self.refine_layers_a = nn.ModuleList([
            FeatureFusionBlock(features, use_bn, up_sample=(i < 4), align_corners=align_corners, fuse=(i!=0))
            for i in range(n_blks)
        ])
        self.output_conv_a = nn.Conv2d(features, out_channels, kernel_size=3, stride=1, padding=1)

        self.refine_layers_b = nn.ModuleList([
            FeatureFusionBlock(features, use_bn, up_sample=(i < 4), align_corners=align_corners, fuse=(i!=0))
            for i in range(n_blks)
        ])
        self.output_conv_b = nn.Conv2d(features, out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, out_features):
        out = []
        for i, x in enumerate(out_features):
            if self.dpt_mode == 'a':
                x = self.projects[i](x)
            x = self.resize_layers[i](x)
            x = self.rn_layers[i](x)
            out.append(x)

        path_a = self.refine_layers_a[0](out[-1])
        path_b = self.refine_layers_b[0](out[-1])
        for i in range(1, len(self.refine_layers_a)):
            path_a = self.refine_layers_a[i](path_a, out[-(i+1)])
            path_b = self.refine_layers_b[i](path_b, out[-(i+1)])
        
        out_a = self.output_conv_a(path_a)
        out_b = self.output_conv_b(path_b)
        out_a = F.interpolate(out_a, scale_factor=2.0, mode="bilinear", align_corners=self.align_corners)
        out_b = F.interpolate(out_b, scale_factor=2.0, mode="bilinear", align_corners=self.align_corners)
        return out_a, out_b