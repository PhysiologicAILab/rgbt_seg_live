import torch
import torch.nn as nn
import torch.functional as F
# from models.mobilenetv2 import mobilenet_v2
# from models.resnet import Backbone_ResNet50_in3
from models.Res2Net_v1b import res2net50_v1b_26w_4s

criterion_MAE = nn.L1Loss().cuda()

def loss_fn(x, y):
	x = F.normalize(x, dim=1, p=2)
	y = F.normalize(y, dim=1, p=2)
	return 2 - 2 * (x * y).sum(dim=1)

class mutual_channel(nn.Module):



    def __init__(self, num_channels,deep_channels,reduction_ratio=2):

        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """

        super(mutual_channel, self).__init__()

        
        num_channels_reduced = 128

        self.reduction_ratio = reduction_ratio

        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)

        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        
        
        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()



    def forward(self, input_tensor,rgb):

        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """      
        #deep = torch.nn.functional.upsample(deep, size=input_tensor.size()[2:], mode='bilinear')

        batch_size, num_channels, H, W = input_tensor.size()

   
        squeeze_tensor = rgb.view(batch_size, num_channels, -1).mean(dim=2)
        #squeeze_tensor1 = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)


        fc_out_1 = self.relu(self.fc1(squeeze_tensor))

        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

       
        a, b = squeeze_tensor.size()
        
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1)) + rgb
        

        return output_tensor


class ASPP(nn.Module): # deeplab

    def __init__(self, dim,in_dim):
        super(ASPP, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim,in_dim , 3,padding=1),nn.BatchNorm2d(in_dim),
             nn.PReLU())
        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
         )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1),nn.BatchNorm2d(down_dim),  nn.PReLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )

    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = torch.nn.functional.upsample(self.conv5(torch.nn.functional.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear')
        return self.fuse(torch.cat((conv1, conv2, conv3,conv4, conv5), 1))

def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )
    
class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
        
class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1, visual = 1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1)),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
                )

        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0,x1,x2),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out
        
        
        
class MAM(nn.Module):
    def __init__(self,current_channels,deep_channels):
        super(MAM, self).__init__()
        self.stn = convblock(deep_channels, 64, 1, 1, 0)
        #self.stn = BasicRFB(64,64)
        self.fus1 = convblock(current_channels, 64, 1, 1, 0)
        self.alpha = nn.Conv2d(64, 1, 1, 1, 0)
        self.beta = nn.Conv2d(current_channels, 1, 1, 1, 0)
        #self.fus2 = convblock(128, 64, 1, 1, 0)
        #self.relu = nn.ReLU()


    def forward(self, rgb, thermal,deep):

        deep_refine = torch.nn.functional.interpolate(self.stn(deep), thermal.size()[2:], mode="bilinear")
        #in1 = self.fus1(torch.cat([gr, gt],dim=1))

        #affine_gt = self.alpha(deep_refine)*gt + self.bata(gr)
        pseudo_gt = torch.sigmoid(self.alpha(deep_refine))
        rgb_q = torch.sigmoid(self.beta(thermal))
        inter = (pseudo_gt * rgb_q).sum(dim=(2, 3))
        union = (pseudo_gt + rgb_q).sum(dim=(2, 3))
        IOU = (inter + 1e-6) / (union - inter + 1e-6)
        IOU = IOU.unsqueeze(1).unsqueeze(1)
        #in1 = self.fus1(gr+affine_gt)
        fuse1 = rgb  + (IOU) * thermal
        #in2 = self.fus2(torch.cat((in1 ,deep_refine),1))
        
        return self.fus1(fuse1) +deep_refine

        #in3 = self.fus3(torch.cat((gt * filter ,gr),1))
        #return self.combine(torch.cat([in1,in2,in3],dim=1))
        
        
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi
        
        
class BasicConv2d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
    ):
        super(BasicConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)
        
class DenseLayer(nn.Module):
    def __init__(self, in_C, out_C, down_factor=4, k=4):
        """
        
        
        """
        super(DenseLayer, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        self.down = nn.Conv2d(in_C, mid_C, 1)
        #self.att = Attention_block(in_C,in_C,in_C)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            
            self.denseblock.append(BasicConv2d(mid_C * i, mid_C, 3, 1, 1))
            #self.denseblock.append(Attention_block(mid_C * i,in_C,mid_C))

        self.fuse = BasicConv2d(in_C + mid_C, out_C, kernel_size=3, stride=1, padding=1)

    def forward(self, in_feat):
        #in_feat = self.att(in_feat,current)
        down_feats = self.down(in_feat)
        #down_feats = self.att(down_feats,current)
        out_feats = []
        for denseblock in self.denseblock:
            feats = denseblock(torch.cat((*out_feats, down_feats), dim=1))
            out_feats.append(feats)
        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)


class LSNet(nn.Module):
    def __init__(self):
        super(LSNet, self).__init__()
        # rgb,depth encode
        #self.rgb_pretrained = mobilenet_v2()
        '''
        (
            self.layer1_rgb,
            self.layer2_rgb,
            self.layer3_rgb,
            self.layer4_rgb,
            self.layer5_rgb,
        ) = Backbone_ResNet50_in3(pretrained=True)
        '''
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        #self.depth_pretrained = mobilenet_v2()

        # Upsample_model
        
        
        self.rgbconv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.rgbconv2 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.rgbconv3 = nn.Conv2d(512, 64, kernel_size=3, padding=1)
        self.rgbconv4 = nn.Conv2d(1024, 64, kernel_size=3, padding=1)
        self.rgbconv5 = nn.Conv2d(2048, 64, kernel_size=3, padding=1)
        

        # Upsample_model
        
        self.upsample1_g = nn.Sequential(nn.Conv2d(128, 32, 3, 1, 1, ), nn.BatchNorm2d(32), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=4, ))

        self.upsample2_g = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1, ), nn.BatchNorm2d(64), nn.GELU(),
                                         )

        self.upsample3_g = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1, ), nn.BatchNorm2d(64), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample4_g = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1, ), nn.BatchNorm2d(64), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))
        

        self.upsample5_g = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, ), nn.BatchNorm2d(64), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))


        self.conv_g = nn.Conv2d(32, 3, 1)
        self.convr_g = nn.Conv2d(2048, 3, 1)
        #self.convt_g = nn.Conv2d(2048, 1, 1)



    def forward(self, rgb, ti):
        # rgb
        #ti = torch.cat((ti,ti,ti),1)
        
        A1 = self.resnet.conv1(rgb)
        A1 = self.resnet.bn1(A1)
        A1 = self.resnet.relu(A1)
        A1 = self.resnet.maxpool(A1)      # bs, 64, 88, 88
        A2 = self.resnet.layer1(A1)      # bs, 256, 88, 88
        A3 = self.resnet.layer2(A2)     # bs, 512, 44, 44
        A4 = self.resnet.layer3(A3)     # bs, 1024, 22, 22
        A5 = self.resnet.layer4(A4)
        '''
        A1_t = self.resnet.conv1(ti)
        A1_t = self.resnet.bn1(A1_t)
        A1_t = self.resnet.relu(A1_t)
        A1_t = self.resnet.maxpool(A1_t)      # bs, 64, 88, 88
        A2_t = self.resnet.layer1(A1_t)      # bs, 256, 88, 88
        A3_t = self.resnet.layer2(A2_t)     # bs, 512, 44, 44
        A4_t = self.resnet.layer3(A3_t)     # bs, 1024, 22, 22
        A5_t = self.resnet.layer4(A4_t)
        
        
        F5 = self.channel5(A5_t,A5) #+ A5 #self.channel5(A5,A5_t)

        F4 = self.channel4(A4_t,A4) #+ A4#self.channel4(A4,A4_t)
        F3 = self.channel3(A3_t,A3) #+ A3#self.channel3(A3,A3_t)
        F2 = self.channel2(A2_t,A2) #+ A2 #self.channel2(A2,A2_t)
        F1 = self.channel1(A1_t,A1) #+ A1#self.channel1(A1,A1_t)
        
        F5 = F5 + A5
        F4 = F4 + A4
        F3 = F3 + A3
        F2 = F2 + A2
        F1 = F1 + A1
        
        '''
        
        
        F5 =  A5
        F4 =  A4
        F3 =  A3
        F2 =  A2
        F1 =  A1
        
        
        F5 = self.rgbconv5(F5)
        F4 = self.rgbconv4(F4)
        F3 = self.rgbconv3(F3)
        F2 = self.rgbconv2(F2)
        F1 = self.rgbconv1(F1)
        


        
        


        F5 = self.upsample5_g(F5)

        F4 = torch.cat((F4, F5), dim=1)     
        F4 = self.upsample4_g(F4)

        F3 = torch.cat((F3, F4), dim=1)
        F3 = self.upsample3_g(F3)

        F2 = torch.cat((F2, F3), dim=1)
        F2 = self.upsample2_g(F2)

        F1 = torch.cat((F1, F2), dim=1)
        F1 = self.upsample1_g(F1)
        
        
        
        out = torch.nn.functional.interpolate(self.conv_g(F1), rgb.size()[2:], mode="bilinear")

        
        if self.training:
            out_a5 = torch.nn.functional.interpolate(self.convr_g(A5), rgb.size()[2:], mode="bilinear")
            
      

            return out,out_a5
            

        return out
