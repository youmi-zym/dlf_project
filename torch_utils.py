import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Dict, Tuple
import math


class Conv2d(nn.Module):
    def __init__(self,
                in_planes, 
                out_planes,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                bias=False,
                norm=None,
                activation=None,):
            super(Conv2d, self).__init__()

            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, bias=bias)
            if norm is not None:
                self.bn = nn.BatchNorm2d(out_planes)
            else:
                self.bn = nn.Identity()

            if activation is not None:
                self.activation = nn.ReLU(inplace=True)
            else:
                self.activation = nn.Identity()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class Conv3d(nn.Module):
    def __init__(self,
                in_planes, 
                out_planes,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                bias=False,
                norm=None,
                activation=None,):
            super(Conv3d, self).__init__()

            self.conv = nn.Conv3d(in_planes, out_planes, kernel_size, stride, padding, dilation, bias=bias)
            if norm is not None:
                self.bn = nn.BatchNorm3d(out_planes)
            else:
                self.bn = nn.Identity()

            if activation is not None:
                self.activation = nn.ReLU(inplace=True)
            else:
                self.activation = nn.Identity()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class ConvTranspose3d(nn.Module):
    def __init__(self,
                in_planes, 
                out_planes,
                kernel_size=1,
                stride=1,
                padding=0,
                output_padding=0,
                dilation=1,
                bias=False,
                norm=None,
                activation=None,):
            super(ConvTranspose3d, self).__init__()

            self.conv = nn.ConvTranspose3d(in_planes, out_planes, kernel_size, stride, padding, dilation, bias=bias)
            if norm is not None:
                self.bn = nn.BatchNorm3d(out_planes)
            else:
                self.bn = nn.Identity()

            if activation is not None:
                self.activation = nn.ReLU(inplace=True)
            else:
                self.activation = nn.Identity()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, out_planes, stride, downsample, padding, dilation, norm, activation):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(in_planes,  out_planes, 3, stride, padding, dilation, bias=False, norm=(norm, out_planes), activation=activation)
        self.conv2 = Conv2d(out_planes, out_planes, 3, 1,      padding, dilation, bias=False, norm=(norm, out_planes), activation=None)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        if self.downsample is not None:
            x = self.downsample(x)
        out += x

        return out


class Feature(nn.Module):
    def __init__(self,
                 in_planes:int = 3,
                 norm: str = 'BN',
                 activation: Union[str, list, tuple] = 'ReLU'):
        super(Feature, self).__init__()

        self.in_planes              = in_planes
        self.norm                   = norm
        self.activation             = activation

        self.firstconv = nn.Sequential(
            Conv2d(in_planes, 32, 3, 2, 1, 1, bias=False, norm=(norm, 32), activation=activation),
            Conv2d(32,        32, 3, 1, 1, 1, bias=False, norm=(norm, 32), activation=activation),
            Conv2d(32,        32, 3, 1, 1, 1, bias=False, norm=(norm, 32), activation=activation),
        )

        # For building Basic Block
        self.in_planes = 32

        self.layer1 = self._make_layer(BasicBlock, 32,  3, 1, 1, 1, norm=norm, activation=activation)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1, norm=norm, activation=activation)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1, norm=norm, activation=activation)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 2, 2, norm=norm, activation=activation)

        self.branch1 = nn.Sequential(
            nn.AvgPool2d((64, 64), stride=(64, 64)),
            Conv2d(128, 32, 1, 1, 0, 1, bias=False, norm=(norm, 32), activation=activation),
        )
        self.branch2 = nn.Sequential(
            nn.AvgPool2d((32, 32), stride=(32, 32)),
            Conv2d(128, 32, 1, 1, 0, 1, bias=False, norm=(norm, 32), activation=activation),
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d((16, 16), stride=(16, 16)),
            Conv2d(128, 32, 1, 1, 0, 1, bias=False, norm=(norm, 32), activation=activation),
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d((8, 8), stride=(8, 8)),
            Conv2d(128, 32, 1, 1, 0, 1, bias=False, norm=(norm, 32), activation=activation),
        )
        self.lastconv = nn.Sequential(
            Conv2d(320, 128, 1, 1, 0, 1, bias=False, norm=(norm, 128), activation=activation),
            nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, dilation=1, bias=False)
        )

        self.weight_init()

    def _make_layer(self, block, out_planes, blocks, stride, padding, dilation, norm, activation):
        downsample = None
        if stride != 1 or self.in_planes != out_planes * block.expansion:
            downsample = Conv2d(
                self.in_planes, out_planes * block.expansion,
                kernel_size=1, stride=stride, padding=0, dilation=1,
                norm=(norm, out_planes * block.expansion), activation=None
            )

        layers = []
        layers.append(
            block(self.in_planes, out_planes, stride, downsample, padding, dilation, norm=norm, activation=activation)
        )
        self.in_planes = out_planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.in_planes, out_planes, 1, None, padding, dilation, norm=norm, activation=activation)
            )

        return nn.Sequential(*layers)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _forward(self, x):
        # [B, 32, H//2, W//2]
        output_2_0 = self.firstconv(x)
        # [B, 32, H//2, W//2]
        output_2_1 = self.layer1(output_2_0)
        # [B, 64, H//4, W//4]
        output_4_0 = self.layer2(output_2_1)
        # [B, 128, H//4, W//4]
        output_4_1 = self.layer3(output_4_0)
        # [B, 128, H//4, W//4]
        output_8 = self.layer4(output_4_1)

        output_branch1 = self.branch1(output_8)
        output_branch1 = F.interpolate(
            output_branch1, (output_8.size()[2], output_8.size()[3]),
            mode='bilinear', align_corners=True
        )

        output_branch2 = self.branch2(output_8)
        output_branch2 = F.interpolate(
            output_branch2, (output_8.size()[2], output_8.size()[3]),
            mode='bilinear', align_corners=True
        )

        output_branch3 = self.branch3(output_8)
        output_branch3 = F.interpolate(
            output_branch3, (output_8.size()[2], output_8.size()[3]),
            mode='bilinear', align_corners=True
        )

        output_branch4 = self.branch4(output_8)
        output_branch4 = F.interpolate(
            output_branch4, (output_8.size()[2], output_8.size()[3]),
            mode='bilinear', align_corners=True
        )

        output_feature = torch.cat(
            (output_4_0, output_8, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature

    def forward(self, *input):
        if len(input) != 2:
            raise ValueError('expected input length 2 (got {} length input)'.format(len(input)))

        l_img, r_img = input

        l_fms = self._forward(l_img)
        r_fms = self._forward(r_img)

        return l_fms, r_fms


class Aggregation(nn.Module):
    def __init__(self,
                 in_planes: int,
                 num_hourglass: int = 3,
                 norm: str = 'BN',
                 activation: Union[str, List, Tuple] = 'ReLU'):
        super(Aggregation, self).__init__()
        self.in_planes = in_planes
        self.num_hourglass = num_hourglass
        self.norm = norm
        self.activation = activation

        self.init = nn.Sequential(
            Conv3d(in_planes, 32, 3, 1, 1, bias=True, norm=(norm, 32), activation=activation),
            Conv3d(32,        32, 3, 1, 1, bias=False, norm=(norm, 32), activation=None),
        )
        self.init_residual = nn.Sequential(
            Conv3d(32, 32, 3, 1, 1, bias=False, norm=(norm, 32), activation=activation),
            Conv3d(32, 32, 3, 1, 1, bias=False, norm=(norm, 32), activation=None),
        )

        self.residual = nn.ModuleList()
        self.classifier = nn.ModuleList()

        for i in range(num_hourglass):
            self.residual.append(
                Hourglass(32, norm, activation),
            )

            self.classifier.append(
                nn.Sequential(
                    Conv3d(32, 32, 3, 1, 1, bias=False, norm=(norm, 32), activation=activation),
                    nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
                )
            )

        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, raw_cost:torch.Tensor, to_full:bool=True):
        B, C, D, H, W = raw_cost.shape

        # raw_cost: (BatchSize, Channels*2, MaxDisparity/4, Height/4, Width/4)
        init_cost = self.init(raw_cost)
        init_cost = self.init_residual(init_cost) + init_cost

        out, pre, post = init_cost, None, None
        costs = []
        for i in range(self.num_hourglass):
            out, pre, post = self.residual[i](out, pre, post)
            cost = out + init_cost
            cost = self.classifier[i](cost)
            if i > 0:
                cost = cost + costs[-1]
            costs.append(cost)

        if to_full:
            full_d, full_h, full_w = D*4, H*4, W*4
            align_corners = True
            for i, cost in enumerate(costs):
                # (BatchSize, 1, MaxDisparity, Height, Width)
                cost = F.interpolate(cost, size=(full_d, full_h, full_w), mode='trilinear', align_corners=align_corners)
                # (BatchSize, max_disp, Height, Width)
                costs[i] = cost.squeeze(dim=1)
        else:
            for i, cost in enumerate(costs):
                costs[i] = cost.squeeze(dim=1)


        # make the best at the front
        costs.reverse()

        return costs


class Hourglass(nn.Module):
    def __init__(self,
                 in_planes: int,
                 norm: str = 'BN3d',
                 activation: Union[str, Tuple, List] = 'ReLU'):
        super(Hourglass, self).__init__()
        self.in_planes = in_planes
        self.norm = norm
        self.activation = activation

        self.conv1 = Conv3d(
            in_planes, in_planes * 2,
            kernel_size=3, stride=2, padding=1, bias=False,
            norm=(norm, in_planes * 2), activation = activation
        )

        self.conv2 = Conv3d(
            in_planes * 2, in_planes * 2,
            kernel_size=3, stride=1, padding=1, bias=False,
            norm = (norm, in_planes * 2), activation = None
        )

        self.conv3 = Conv3d(
            in_planes * 2, in_planes * 2,
            kernel_size=3, stride=2, padding=1, bias=False,
            norm=(norm, in_planes * 2), activation = activation
        )

        self.conv4 = Conv3d(
            in_planes * 2, in_planes * 2,
            kernel_size=3, stride=1, padding=1, bias=False,
            norm=(norm, in_planes * 2), activation=activation
        )
        self.conv5 = ConvTranspose3d(
            in_planes * 2, in_planes * 2,
            kernel_size=3, padding=1, output_padding=1, stride=2, bias=False,
            norm=(norm, in_planes * 2), activation=None

        )
        self.conv6 = ConvTranspose3d(
            in_planes * 2, in_planes,
            kernel_size=3, padding=1, output_padding=1, stride=2, bias=False,
            norm=(norm, in_planes), activation=None

        )

    def forward(self, x, presqu=None, postsqu=None):
        # in: [B, C, D, H, W], out: [B, 2C, D/2, H/2, W/2]
        out = self.conv1(x)
        # in: [B, 2C, D/2, H/2, W/2], out: [B, 2C, D/2, H/2, W/2]
        pre = self.conv2(out)
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        # in: [B, 2C, D/2, H/2, W/2], out: [B, 2C, D/4, H/4, W/4]
        out = self.conv3(pre)
        # in: [B, 2C, D/4, H/4, W/4], out: [B, 2C, D/4, H/4, W/4]
        out = self.conv4(out)

        # in: [B, 2C, D/4, H/4, W/4], out: [B, 2C, D/2, H/2, W/2]
        if presqu is not None:
            D, H, W = presqu.shape[-3:]
            out = self.conv5(out)
            out = F.interpolate(out, size=(D, H, W), mode='trilinear', align_corners=True)
            post = F.relu(out + presqu, inplace=True)
        else:
            D, H, W = pre.shape[-3:]
            out = self.conv5(out)
            out = F.interpolate(out, size=(D, H, W), mode='trilinear', align_corners=True)
            post = F.relu(out + pre, inplace=True)

        # in: [B, 2C, D, H/2, W/2], out: [B, C, D, H, W]
        out = self.conv6(post)
        D, H, W = x.shape[-3:]
        out = F.interpolate(out, size=(D, H, W), mode='trilinear', align_corners=True)

        return out, pre, post


class SOFTARGMIN(nn.Module):
    def __init__(self, temperature:float=1.0, normalize:bool=True):
        super(SOFTARGMIN, self).__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, cost_volume, max_disp=192, disp_sample=None):

        # note, cost volume direct represent similarity
        # 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.

        if cost_volume.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(cost_volume.dim()))

        # scale cost volume with temperature
        cost_volume = cost_volume * self.temperature

        if self.normalize:
            prob_volume = F.softmax(cost_volume, dim=1)
        else:
            prob_volume = cost_volume

        if disp_sample is None:
            B, D, H, W = prob_volume.shape
            device = prob_volume.device

            disp_sample = torch.linspace(0, max_disp-1, max_disp).to(device)
            disp_sample = disp_sample[None, :, None, None].repeat(B, 1, H, W).contiguous()

        assert prob_volume.shape == disp_sample.shape, f'The shape of disparity samples {disp_sample.shape} and cost volume {prob_volume.shape} should be consistent!'

        # compute disparity: (BatchSize, 1, Height, Width)
        disp_map = torch.sum(prob_volume * disp_sample, dim=1, keepdim=True)

        return disp_map


def cat_fms(reference_fm, target_fm, max_disp=192, start_disp=0, dilation=1, disp_sample=None):
    device = reference_fm.device
    N, C, H, W = reference_fm.shape

    end_disp = start_disp + max_disp - 1
    disp_sample_number = (max_disp + dilation - 1) // dilation
    disp_index = torch.linspace(start_disp, end_disp, disp_sample_number)

    concat_fm = torch.zeros(N, C * 2, disp_sample_number, H, W).to(device)
    idx = 0
    for i in disp_index:
        i = int(i) # convert torch.Tensor to int, so that it can be index
        if i > 0:
            concat_fm[:, :C, idx, :, i:] = reference_fm[:, :, :, i:]
            concat_fm[:, C:, idx, :, i:] = target_fm[:, :, :, :-i]
        elif i == 0:
            concat_fm[:, :C, idx, :, :] = reference_fm
            concat_fm[:, C:, idx, :, :] = target_fm
        else:
            concat_fm[:, :C, idx, :, :i] = reference_fm[:, :, :, :i]
            concat_fm[:, C:, idx, :, :i] = target_fm[:, :, :, abs(i):]
        idx = idx + 1

    concat_fm = concat_fm.contiguous()

    return concat_fm
