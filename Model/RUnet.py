import torch
import torch.nn as nn
import torch.nn.functional as F

basic_dims = 8
num_modals = 1
patch_size = [2, 8, 8]


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        #m = nn.InstanceNorm3d(planes)
        m = nn.GroupNorm(4, planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m


class general_conv3d_prenorm(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', is_training=True,
                 act_type='lrelu', relufactor=0.2):
        super(general_conv3d_prenorm, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride,
                              padding=padding, padding_mode=pad_type, bias=True)

        self.norm = normalization(in_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)

    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        return x


class fusion_prenorm(nn.Module):
    def __init__(self, in_channel=64, num_cls=1):
        super(fusion_prenorm, self).__init__()
        self.fusion_layer = nn.Sequential(
            general_conv3d_prenorm(in_channel * num_modals, in_channel, k_size=1, padding=0, stride=1),
            general_conv3d_prenorm(in_channel, in_channel, k_size=3, padding=1, stride=1),
            general_conv3d_prenorm(in_channel, in_channel, k_size=1, padding=0, stride=1))

    def forward(self, x):
        return self.fusion_layer(x)


class Encoder(nn.Module):
    def __init__(self, flag=True):
        super(Encoder, self).__init__()
        if flag:
            self.e1_c1 = nn.Conv3d(in_channels=1, out_channels=basic_dims, kernel_size=3, stride=1, padding=1,
                                   padding_mode='zeros', bias=True)
        else:
            self.e1_c1 = nn.Conv3d(in_channels=2, out_channels=basic_dims, kernel_size=3, stride=1, padding=1,
                                   padding_mode='zeros', bias=True)
        self.e1_c2 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='zeros')
        self.e1_c3 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='zeros')

        self.e2_c1 = general_conv3d_prenorm(basic_dims, basic_dims * 2, stride=2, pad_type='zeros')
        self.e2_c2 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, pad_type='zeros')
        self.e2_c3 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, pad_type='zeros')

        self.e3_c1 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 4, stride=2, pad_type='zeros')
        self.e3_c2 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, pad_type='zeros')
        self.e3_c3 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, pad_type='zeros')

        self.e4_c1 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 8, stride=2, pad_type='zeros')
        self.e4_c2 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, pad_type='zeros')
        self.e4_c3 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, pad_type='zeros')

        self.e5_c1 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 16, stride=2, pad_type='zeros')
        self.e5_c2 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 16, pad_type='zeros')
        self.e5_c3 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 16, pad_type='zeros')

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))

        return x1, x2, x3, x4, x5


class Decoder_fuse(nn.Module):
    def __init__(self, num_cls=1):
        super(Decoder_fuse, self).__init__()

        self.d4_c1 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='zeros')
        self.d4_c2 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='zeros')
        self.d4_out = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, k_size=1, padding=0, pad_type='zeros')

        self.d3_c1 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='zeros')
        self.d3_c2 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='zeros')
        self.d3_out = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type='zeros')

        self.d2_c1 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='zeros')
        self.d2_c2 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='zeros')
        self.d2_out = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type='zeros')

        self.d1_c1 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type='zeros')
        self.d1_c2 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type='zeros')
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='zeros')

        self.seg_d4 = nn.Conv3d(in_channels=basic_dims * 16, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                bias=True)
        self.seg_d3 = nn.Conv3d(in_channels=basic_dims * 8, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                bias=True)
        self.seg_d2 = nn.Conv3d(in_channels=basic_dims * 4, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                bias=True)
        self.seg_d1 = nn.Conv3d(in_channels=basic_dims * 2, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                bias=True)
        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.RFM5 = fusion_prenorm(in_channel=basic_dims * 16, num_cls=num_cls)
        self.RFM4 = fusion_prenorm(in_channel=basic_dims * 8, num_cls=num_cls)
        self.RFM3 = fusion_prenorm(in_channel=basic_dims * 4, num_cls=num_cls)
        self.RFM2 = fusion_prenorm(in_channel=basic_dims * 2, num_cls=num_cls)
        self.RFM1 = fusion_prenorm(in_channel=basic_dims * 1, num_cls=num_cls)
        self.act = nn.LeakyReLU(0.2,True)

    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.RFM5(x5)
        de_x5 = self.d4_c1(self.up2(de_x5))
        de_x4 = self.RFM4(x4)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        de_x4 = self.d3_c1(self.up2(de_x4))

        de_x3 = self.RFM3(x3)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        de_x2 = self.RFM2(x2)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        de_x1 = self.RFM1(x1)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)

        pred = self.act(logits)

        return pred, de_x5


class RUnet(nn.Module):
    def __init__(self, num_cls=3):
        super(RUnet, self).__init__()
        self.ADC_encoder = Encoder(flag=True)
        self.decoder_fuse = Decoder_fuse(num_cls=num_cls)

        self.is_training = True

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):

        ADC_x1, ADC_x2, ADC_x3, ADC_x4, ADC_x5 = self.ADC_encoder(x[:, 0:1, :, :, :])


        x1 = ADC_x1
        x2 = ADC_x2
        x3 = ADC_x3
        x4 = ADC_x4
        x5 = ADC_x5

        fuse_pred,de_x5 = self.decoder_fuse(x1, x2, x3, x4, x5)

        return fuse_pred, ADC_x1, ADC_x2, ADC_x3, ADC_x4, ADC_x5
