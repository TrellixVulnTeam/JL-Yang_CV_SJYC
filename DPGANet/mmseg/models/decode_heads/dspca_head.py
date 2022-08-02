import torch
from torch.functional import split
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from torch import nn
from torch.nn import Module,  Conv2d,Parameter, Softmax
from mmseg.ops import resize
from mmseg.core import add_prefix
from ..builder import HEADS
from .decode_head import BaseDecodeHead

class ASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, dilations, size ,in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg , align_corners ):
        super(ASPPModule, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners =  align_corners
        for dilation in dilations:
            self.append(
                ConvModule(
                    self.in_channels,
                    self.channels//4,
                    1 if dilation == 1 else 3,
                    dilation=dilation,
                    padding=0 if dilation == 1 else dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        


    def forward(self, x):
        """Forward function."""

        
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))
        
        
        aspp_outs = torch.cat(aspp_outs, dim=1)
       
        return aspp_outs

class IPAM_Module(Module):
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(IPAM_Module, self).__init__()
        self.chanel_in = in_dim
 
        # 先经过3个卷积层生成3个新特征图B C D （尺寸不变）
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))  # α尺度系数初始化为0，并逐渐地学习分配到更大的权重
 
        self.softmax = Softmax(dim=-1)  # 对每一行进行softmax
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B × C × H × W)
            returns :
                out : attention value + input feature
                attention: B × (H×W) × (H×W)
        """
        m_batchsize, C, height, width = x.size()
        # B -> (N,C,HW) -> (N,HW,C)
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        # C -> (N,C,HW)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        # BC，空间注意图 -> (N,HW,HW)
        energy = torch.bmm(proj_query, proj_key)
        # S = softmax(BC) -> (N,HW,HW)
        attention = self.softmax(energy)
        
        #接下来进行judge操作
        #通过β筛选出需要留下的位置——S
        # b = torch.tensor(0.5)
        # b = b.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # s = torch.gt(attention,b)
        s = torch.gt(attention,0.5)
        attention = torch.mul(s,attention)
        # D -> (N,C,HW)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        # DS -> (N,C,HW)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # torch.bmm表示批次矩阵乘法
        # output -> (N,C,H,W)
        out = out.view(m_batchsize, C, height, width)
 
        out = self.gamma*out + x
        return out
 

class CAM(Module):
    """Channel Attention Module (CAM)"""

    def __init__(self):
        super(CAM, self).__init__()
        self.gamma = Scale(0)

    def forward(self, x):
        """Forward function."""
        batch_size, channels, height, width = x.size()
        proj_query = x.view(batch_size, channels, -1)
        proj_key = x.view(batch_size, channels, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy_new, dim=-1)
        proj_value = x.view(batch_size, channels, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, channels, height, width)

        out = self.gamma(out) + x
        return out

#xzl 11-10
class R2(nn.Module):
    def __init__(self, inplanes, planes , stride=1 , scale = 4):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(R2, self).__init__()

        sub_channel = int(planes/scale)
        
        self.conv1 = nn.Conv2d(inplanes, sub_channel, kernel_size=1, stride=1,bias=False)
        self.bn1 = nn.BatchNorm2d( sub_channel )
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale 
  
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d( int(sub_channel/4)  , int(sub_channel/4) , kernel_size=3, stride = stride, padding=1, bias=False))
          bns.append(nn.BatchNorm2d( int(sub_channel/4) )) 
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns) 

        self.conv3 = nn.Conv2d( sub_channel , planes, kernel_size=1 , bias=False)
        self.bn3 = nn.BatchNorm2d(planes )

        self.relu = nn.ReLU(inplace=True)
        self.scale = scale
        
        self.sub_channel = sub_channel

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print(out.size())

        sub_c = int(out.size(1)/4)
        # print(sub_c)
        spx = torch.split(out, sub_c , dim=1 )
        # print(spx)
        # print(len(spx))
        # print(self.nums)
        
        for i in range(0 , self.nums):
            # print(i)
            if i==0 :
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i==0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        # if self.scale != 1 :
        #   out = torch.cat(( out, spx[self.nums-1] ),1)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

class dsm_block(nn.Module):
    def __init__(self, inplanes , out_c ,scale):
        super( dsm_block , self).__init__()

        self.scale = scale
        self.flow = nn.Sequential(
            nn.Conv2d( inplanes ,  out_c  , kernel_size=3 , stride=1 , padding=1 ,bias=False),
            R2(  out_c  ,  out_c  , stride=1 , scale=self.scale ),
            nn.Conv2d(  out_c  ,  out_c  , kernel_size=3 , stride=1 , padding=1 ,bias=False),
            nn.Conv2d(  out_c  ,  out_c  , kernel_size=3 , padding=1 ,groups=  out_c  , stride=2 ,bias=False)
    
        )
        
    def forward(self,x):
        return self.flow(x)


class DSMPCF(nn.Module):
    def __init__(self, inplanes ,scale):
        super( DSMPCF, self).__init__()

        self.cutConv = nn.Conv2d(  inplanes  ,  64  , kernel_size=3 , padding=1   , stride=2 ,bias=False)

        self.block1 = dsm_block(64,128,scale)
        self.block2 = dsm_block(128,256,scale)
        self.block3 = dsm_block(256,512,scale)
        self.block4 = dsm_block(512,1024,scale)

    def forward(self,x):
        

        x = self.cutConv(x)

        output = []

        output1 = self.block1(x)
        output.append(output1)

        output2 = self.block2(output1)
        output.append(output2)

        output3 = self.block3(output2)
        output.append(output3)

        output4 = self.block4(output3)
        output.append(output4)

        return output

#xzl 11-10 

# xzl 11-19
class LargefieldUpsampleConnection(nn.Module):
    def __init__(self, in_places , option ,factor1 , factor2 , eps=1e-6):
        super().__init__()
        
        self.option = option

        self.funa = nn.Sequential(
            nn.Conv2d(in_places,in_places//factor1,kernel_size=1),
            nn.Conv2d(in_places//factor1,in_places//factor1,kernel_size=3,dilation=6,padding=6),
            nn.ConvTranspose2d(in_places//factor1,in_places//factor1,kernel_size=2,stride=2)
        )

        self.relu = nn.ReLU()
        self.funb = nn.Sequential(
            nn.Conv2d(in_places//factor1,in_places//factor2,kernel_size=1),
            nn.Conv2d(in_places//factor2,in_places//factor2,kernel_size=3,dilation=12,padding=12),
            nn.ConvTranspose2d(in_places//factor2,in_places//factor2,kernel_size=2,stride=2)
        )
        
    def forward(self,x):
        
        if self.option == 1:
            x_out = self.funa(x)
        else: 
            x_out = self.funa(x)
            x_out = self.relu(x_out)
            x_out = self.funb(x_out)
    
        return x_out.contiguous()  

# xzl 11-19

@HEADS.register_module()
class DSPCAHead(BaseDecodeHead):
    

    def __init__(self, ipam_channels ,dilations=(1, 6, 12, 18),**kwargs):
        super(DSPCAHead, self).__init__(**kwargs)
        assert isinstance(dilations, (list, tuple))
        
        self.dsmpcf = DSMPCF( self.in_channels[0] ,scale=4 )
        

        self.dilations = dilations

        self.cam0 = CAM()
        self.ipam0 = IPAM_Module(1024)

        now_c = 1024
        self.block1 = nn.Sequential(
            ASPPModule(
                dilations,
                16,
                now_c,
                now_c,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                align_corners = self.align_corners
            ),
            LargefieldUpsampleConnection( now_c , 1  , 2 , 2 )    
        )

        now_c = 512
        self.cam1 = CAM()
        self.ipam1 = IPAM_Module(now_c)

        self.block2 = nn.Sequential(
            ASPPModule(
                dilations,
                32,
                now_c,
                now_c ,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                align_corners = self.align_corners
            ),
            LargefieldUpsampleConnection( now_c , 1  , 2 , 2 )    
        )

        now_c = 256
        self.cam2 = CAM()
        self.ipam2 = IPAM_Module(now_c)

        self.block3 = nn.Sequential(
            # ASPPModule(
            #     dilations,
            #     64,
            #     now_c,
            #     now_c ,
            #     conv_cfg=self.conv_cfg,
            #     norm_cfg=self.norm_cfg,
            #     act_cfg=self.act_cfg,
            #     align_corners = self.align_corners
            # ),
            LargefieldUpsampleConnection( now_c , 1  , 2 , 2 )    
        )

        now_c = 128
        self.cam3 = CAM()
        self.ipam3 = IPAM_Module(now_c)

        

    def forward(self, inputs):
    
        """Forward function."""
        x = self._transform_inputs(inputs)

        # print(x[0].size())
        # print(x[1].size())
        dsm = x[0].unsqueeze(1)
        dsm_x = self.dsmpcf(dsm)
        

        outputs = dsm_x[3] + x[4]

        
        outputs = self.cam0(outputs) + self.ipam0(outputs)


        outputs = self.block1(outputs) + x[3] + dsm_x[2]

        outputs = self.cam1(outputs) + self.ipam1(outputs)
       
        outputs = self.block2(outputs) + x[2] + dsm_x[1]

        outputs = self.cam2(outputs) + self.ipam2(outputs)

        outputs = self.block3(outputs) + x[1] + dsm_x[0]
        
        outputs = self.cls_seg(outputs)
        # print(outputs.size())
        return outputs
        


