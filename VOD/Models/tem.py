import torch
import torch.nn as nn
import torchvision.models as models

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RF(nn.Module):
    # Revised from: Receptive Field Block Net for Accurate and Fast Object Detection, 2018, ECCV
    # GitHub: https://github.com/ruinmessi/RFBNet
    def __init__(self, in_channel, out_channel):
        super(RF, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )

        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


#  lapdepth中的图像切割数据
        # new_h = 352
        # new_w = org_w * (352.0/org_h)
        # new_w = int((new_w//16)*16)
#ASPP with Dilated Convolution 还有其他的类型aspp具体看是否相同。
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=atrous_rates[0], dilation=atrous_rates[0])
        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=atrous_rates[1], dilation=atrous_rates[1])
        self.conv3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=atrous_rates[2], dilation=atrous_rates[2])
        self.conv3x3_4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=atrous_rates[3], dilation=atrous_rates[3])
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_output = nn.Conv2d(in_channels + 5 * out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        feature_map = x

        # ASPP模块中的并行卷积操作
        branch_1 = self.conv1x1(x)
        branch_2 = self.conv3x3_1(x)
        branch_3 = self.conv3x3_2(x)
        branch_4 = self.conv3x3_3(x)
        branch_5 = self.conv3x3_4(x)
        global_pool = self.global_pooling(x)
        global_pool = global_pool.expand_as(x)  # 保持尺寸与 x 相同
        
        # 将所有分支的输出拼接在一起
        output = torch.cat([branch_1, branch_2, branch_3, branch_4, branch_5, global_pool], dim=1)
        
        # 使用1x1卷积进行特征融合
        output = self.conv1x1_output(output)
        
        return output