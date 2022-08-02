import torch
import torch.nn as nn

from ..builder import HEADS
from .decode_head import BaseDecodeHead

def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))

class SharedSpatialAttention(nn.Module):
    """Position linear attention"""
    def __init__( self, in_places, eps=1e-6 ):
        super().__init__() #初始化父类
        self.gamma = nn.Parameter(torch.zeros(1)) #可学习输出偏置
        self.in_places = in_places #输入 channel
        self.l2_norm = l2_norm # L2范数
        self.eps = eps #防 nan 参数
        #QKV生成卷积
        self.query_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // 8, kernel_size=1) 
        self.key_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)
        Q = self.l2_norm(Q).permute(-3, -1, -2) #对Q进行L2正则
        K = self.l2_norm(K) #对K进行L2正则
        tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps)) #下方全部
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)
        matrix = torch.einsum('bmn, bcn->bmc', K, V) 
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix) #上方全部
        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (x + self.gamma * weight_value)

class SharedChannelAttention(nn.Module):
    """Channel linear attention"""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.l2_norm = l2_norm
        self.eps = eps

    def forward(self, x):
        batch_size, channels, width, height = x.shape
        Q = x.view(batch_size, channels, -1)
        K = x.view(batch_size, channels, -1)
        V = x.view(batch_size, channels, -1)

        Q = self.l2_norm(Q)
        K = self.l2_norm(K).permute(-3, -1, -2)

        tailor_sum = 1 / (width * height + torch.einsum("bnc, bn->bc", K, torch.sum(Q, dim=-2) + self.eps))
        
        value_sum = torch.einsum("bcn->bn", V).unsqueeze(-1).permute(0, 2, 1)
        value_sum = value_sum.expand(-1, channels, width * height)
        matrix = torch.einsum('bcn, bnm->bcm', V, K)
        matrix_sum = value_sum + torch.einsum("bcm, bmn->bcn", matrix, Q)

        weight_value = torch.einsum("bcn, bc->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, channels, height, width)

        return (x + self.gamma * weight_value)

class DownsampleConnection(nn.Module):


    def __init__(self, in_places ,eps=1e-6):
        super().__init__()
        
        
        self.fun1 = nn.Sequential(
            nn.Conv2d(in_places,in_places*2,kernel_size=1),
            nn.BatchNorm2d(in_places*2),
            nn.Conv2d(in_places*2,in_places*2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_places*2),
            nn.Conv2d(in_places*2,in_places*2,kernel_size=3,stride=2,padding=1)
        )
        
        self.fun2 = nn.Sequential(
            nn.Conv2d(in_places,in_places*2,kernel_size=1),
            nn.BatchNorm2d(in_places*2),
            nn.Conv2d(in_places*2,in_places*2,kernel_size=3,stride=2,padding=1)
        )
        
    def forward(self,x):
        batch_size, channels, width, height = x.shape
       
        relu = nn.ReLU()

        x_out = relu( self.fun1(x) + self.fun2(x) )

        return x_out


class LargefieldUpsampleConnection(nn.Module):
    def __init__(self, in_places , eps=1e-6):
        super().__init__()
        
        self.funa = nn.Sequential(
            nn.Conv2d(in_places,in_places//2,kernel_size=1),
            nn.Conv2d(in_places//2,in_places//2,kernel_size=3,dilation=6,padding=6),
            nn.ConvTranspose2d(in_places//2,in_places//2,kernel_size=2,stride=2)
        )

        self.funb = nn.Sequential(
            nn.Conv2d(in_places//2,in_places//4,kernel_size=1),
            nn.Conv2d(in_places//4,in_places//4,kernel_size=3,dilation=12,padding=12),
            nn.ConvTranspose2d(in_places//4,in_places//4,kernel_size=2,stride=2)
        )
        
        
    def forward(self,x):
        batch_size, channels, width, height = x.shape
        
        relu = nn.ReLU()
        
        x_out = relu( self.funb(self.funa(x)) )
    
        return x_out 

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


@HEADS.register_module()
class DCFAM(BaseDecodeHead):
    
    def __init__(self,  **kwargs):
        super().__init__( input_transform='multiple_select', **kwargs )
        self.SSA = SharedSpatialAttention( self.in_channels[3] )
        
        self.SCA = SharedChannelAttention()
       
        self.DC_256_512 = DownsampleConnection(self.in_channels[2] )
        self.DC_512_1024 = DownsampleConnection( self.in_channels[3] )
        self.DC_128_256 = DownsampleConnection( self.in_channels[1] )

        self.LU_256_1024 = LargefieldUpsampleConnection( self.in_channels[4] )
        self.LU_128_512 = LargefieldUpsampleConnection( self.in_channels[3] )

        self.conv1 = nn.Conv2d(self.in_channels[2],self.in_channels[1],kernel_size=1)
        self.U =  nn.functional.interpolate
        
        self.dsmpcf = DSMPCF( self.in_channels[0] ,scale=4 )

        self.dsm_w0 = nn.Parameter(data=torch.tensor([1.0]),requires_grad=True)
        self.dsm_w1 = nn.Parameter(data=torch.tensor([1.0]),requires_grad=True)
        self.dsm_w2 = nn.Parameter(data=torch.tensor([1.0]),requires_grad=True)
        self.dsm_w3 = nn.Parameter(data=torch.tensor([1.0]),requires_grad=True)


    def forward(self, inputs):
        """Forward function."""

        inputs = self._transform_inputs(inputs)

        dsm = inputs[0].unsqueeze(1)
        dsm_x = self.dsmpcf(dsm)

        # print(dsm_x[3].shape)
#         print(self.dsm_w3)
                
        af4 = inputs[4] + self.DC_512_1024( self.SSA( self.DC_256_512( inputs[2]) ) ) + dsm_x[3]*self.dsm_w3
        
        af3 = self.SSA(inputs[3]) + self.DC_256_512( self.SCA( self.DC_128_256(inputs[1])  )) + dsm_x[2]*self.dsm_w2
        
        af2 = self.SCA(inputs[2]) + self.LU_256_1024(af4) #+ dsm_x[1]*self.dsm_w1
        
        
        af1 = inputs[1] + self.U(self.conv1(af2),scale_factor=2,recompute_scale_factor=False) + self.LU_128_512(af3) #+ dsm_x[0]*self.dsm_w0
        output = self.cls_seg(af1)
        '''
        param= list(self.LU_96_384.named_parameters())
        print(param[1])
        print("*************************************************************************")
        '''
        return output
        
        



if __name__ == "__main__":
    x = torch.rand((10, 16, 256, 256), dtype=torch.float)
    
    LC = LargefieldUpsampleConnection()
    DC = DownsampleConnection()

    
    print(x.shape, LC(x).shape )