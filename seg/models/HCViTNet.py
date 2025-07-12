import torch
import torch.nn as nn
import math

from pytorch_wavelets import DWTForward, DWTInverse
from timm.models.layers import trunc_normal_, DropPath

class AttentionGate(nn.Module):
    def __init__(self, in_channels):
        super(AttentionGate, self).__init__()
        # 1x1 卷积，用于处理指导信号 g
        self.W_g = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 1x1 卷积，用于处理输入特征 x
        self.W_x = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 1x1 卷积，用于生成单通道的注意力图
        self.psi = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        """
        :param x: 输入特征图 (例如，来自DWT的某个子带)
        :param g: 指导信号 (例如，来自解码器的高级特征)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi_in = self.relu(g1 + x1)
        psi_out = self.psi(psi_in)
        attention_map = self.sigmoid(psi_out)
        # 将注意力图应用于输入 x
        return x * attention_map


class WARM(nn.Module):
    def __init__(self, in_channels, wave='haar', mode='zero'):
        """
        :param in_channels: 输入特征图 (x_enc, x_dec) 的通道数
        :param wave: 小波基函数
        :param mode: 信号延拓模式
        """
        super(WARM, self).__init__()
        
        # 1. 分解与重构层
        self.dwt = DWTForward(J=1, wave=wave, mode=mode)
        self.idwt = DWTInverse(wave=wave, mode=mode)
        
        # 2. 生成指导信号的下采样层
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 3. 为四个频域子带创建独立的注意力门
        self.attn_ll = AttentionGate(in_channels)
        self.attn_lh = AttentionGate(in_channels)
        self.attn_hl = AttentionGate(in_channels)
        self.attn_hh = AttentionGate(in_channels)

    def forward(self, x_enc, x_dec):
        """
        :param x_enc: 来自编码器的低级特征图
        :param x_dec: 来自解码器的高级特征图 (shape与x_enc相同)
        :return: 精炼后的特征图，shape与x_enc相同
        """
        # 1. 分解: 将编码器特征分解为4个频域子带
        ll, y_h = self.dwt(x_enc)
        # y_h 是一个列表，其中包含一个 (B, C, 3, H/2, W/2) 的张量
        lh, hl, hh = y_h[0][:,:,0], y_h[0][:,:,1], y_h[0][:,:,2]

        # 2. 指导: 将解码器特征下采样以匹配子带的尺寸
        g = self.pool(x_dec)
        
        # 3. 门控: 对每个子带应用注意力门
        ll_refined = self.attn_ll(ll, g)
        lh_refined = self.attn_lh(lh, g)
        hl_refined = self.attn_hl(hl, g)
        hh_refined = self.attn_hh(hh, g)
        
        # 4. 重构: 将精炼后的子带合并，恢复原始尺寸
        y_h_refined = [torch.stack([lh_refined, hl_refined, hh_refined], dim=2)]
        x_refined = self.idwt((ll_refined, y_h_refined))
        
        return x_refined
    

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, groups, kernel_size=7, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, groups=groups,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6())
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=bias, groups=1,stride=1, padding=0),
            norm_layer(out_channels),
            nn.ReLU6())
        
    def forward(self, x):
        short_cut = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = short_cut + x
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    
class MSAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                attn_drop=0., proj_drop=0., max_sr_ratio=1):

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim // 2 * 3, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.act = nn.GELU()
        self.sr_ratio = max_sr_ratio
        
        self.sr1 = nn.Conv2d(dim, dim, kernel_size=self.sr_ratio, stride=self.sr_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.sr2 = nn.Conv2d(dim, dim, kernel_size=self.sr_ratio//2, stride=self.sr_ratio//2)
        self.norm2 = nn.LayerNorm(dim)
        self.sr3 = nn.Conv2d(dim, dim, kernel_size=self.sr_ratio//4, stride=self.sr_ratio//4)
        self.norm3 = nn.LayerNorm(dim)
        
        self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv3 = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.local_conv1 = nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, stride=1, groups=dim//2)
        self.local_conv2 = nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, stride=1, groups=dim//2)
        self.local_conv3 = nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, stride=1, groups=dim//2)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        
        
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()  #torch.Size([1, 64, 128, 128])
        
        x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1))).contiguous()  #torch.Size([1, 256, 64])
        x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1))).contiguous()  #torch.Size([1, 1024, 64])
        x_3 = self.act(self.norm3(self.sr3(x_).reshape(B, C, -1).permute(0, 2, 1))).contiguous()
        
        
        kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous() 
        kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous() 
        kv3 = self.kv3(x_3).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous() 
        k1, v1 = kv1[0], kv1[1] 
        k2, v2 = kv2[0], kv2[1] 
        k3, v3 = kv3[0], kv3[1] 
        
        attn1 = (q[:, :self.num_heads//2] @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)   
        v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C//2).
                                transpose(1, 2).view(B,C//2, H//self.sr_ratio, W//self.sr_ratio)).\
            view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2).contiguous() 
        x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C//2).contiguous()  
        
        attn2 = (q[:, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C//2).
                                transpose(1, 2).view(B, C//2, H*2//self.sr_ratio, W*2//self.sr_ratio)).\
            view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2).contiguous() 
        x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C//2)
        
        attn3 = (q[:, self.num_heads // 2:] @ k3.transpose(-2, -1)) * self.scale
        attn3 = attn3.softmax(dim=-1)
        attn3 = self.attn_drop(attn3)
        v3 = v3 + self.local_conv3(v3.transpose(1, 2).reshape(B, -1, C//2).
                                transpose(1, 2).view(B, C//2, H*4//self.sr_ratio, W*4//self.sr_ratio)).\
            view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2).contiguous() 
        x3 = (attn3 @ v3).transpose(1, 2).reshape(B, N, C//2)

        x = torch.cat([x1,x2,x3], dim=-1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MSFormerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=8):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = MSAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, max_sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B,C,H,W = x.shape
        x = x.permute(0, 2, 3, 1).view(B,H*W,C).contiguous() # B H*W C
        
        x_ = self.attn(self.norm1(x), H, W)
        x = x + self.drop_path(x_)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        x = x.view(B,H,W,C).permute(0, 3, 1, 2).contiguous() # B C H W
        return x 


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x) # B C H W
        x = self.norm(x)
        return x


class DownSample(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous() # B H W C
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = self.norm(x)
        x = self.reduction(x).permute(0, 3, 1, 2).contiguous() # B 4*C H/2 W/2
        return x 


class UpSample(nn.Module):
    def __init__(self,dim,factor=2):
        super(UpSample,self).__init__()
        self.up_factor = factor
        self.up = nn.Sequential(
            nn.Conv2d(dim, self.up_factor*dim, kernel_size=1,stride=1,padding=0),
            nn.PixelShuffle(self.up_factor)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class HCViTNet(nn.Module):
    def __init__(self, img_size=256, in_chans=3, num_classes=2,
                 embed_dim=16, depths=[2, 2, 2, 2], mlp_ratio=4., drop_rate=0.):
        super().__init__()
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        
        self.patch_embed = PatchEmbed(img_size, 2, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim,self.patch_embed.grid_size[0],self.patch_embed.grid_size[1]))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # encoder
        self.encoder = nn.ModuleList()
        for i in range(self.num_layers):
            layer = nn.ModuleList([
                nn.Sequential(
                    ConvBNReLU(embed_dim * 2**i, embed_dim * 2**i, groups=embed_dim * 2**i),
                    MSFormerBlock(
                    dim=embed_dim * 2**i, num_heads=2,
                    mlp_ratio=mlp_ratio,
                    sr_ratio=8))
                for _ in range(depths[i])
            ])
            self.encoder.append(layer)
            
            if i != self.num_layers-1:
                self.encoder.append(DownSample(embed_dim * 2**i))
        
        # decoder
        self.decoder = nn.ModuleList()
        self.fusion = nn.ModuleList()
        for i in reversed(range(self.num_layers-1)):
            #print(i) # 3 2 1 0
            self.decoder.append(UpSample(embed_dim * 2**(i+1))) # upsample
            #if i != self.num_layers:
            layer = nn.ModuleList([
                nn.Sequential(
                    ConvBNReLU(embed_dim * 2**i, embed_dim * 2**i, groups=embed_dim * 2**i),
                    MSFormerBlock(
                    dim=embed_dim * 2**i,
                    num_heads=2,
                    mlp_ratio=mlp_ratio,
                    sr_ratio=8)
                ) for _ in range(depths[i])
            ])
            self.decoder.append(layer)



        self.fusion64 = WARM(64)
        self.fusion32 = WARM(32)
        self.fusion16 = WARM(16)

        # output head
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        # patch_embed
        x = self.patch_embed(x) 
        x = x + self.pos_embed
        x = self.pos_drop(x)    # B C H W

        # encoder
        skips = [x]
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, DownSample):
                x = layer(x)
                skips.append(x)
            else:
                for blk in layer:
                    x = blk(x)
        # decoder
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, UpSample): 
                x = layer(x)
                if i == 0:
                    x = x + skips[2]
                elif i == 2:
                    x = x+skips[1]
                elif i == 4:
                    x = self.fusion16(x,skips[0]) + x
            else:
                for blk in layer:
                    x = blk(x)
        
        # output head
        x = self.head(x)
        return x


if __name__ == "__main__":

    model = HCViTNet(256)
    x = torch.rand((1,3,256,256))
    output = model(x)
    print(output.shape)
    
    if 1:
        from fvcore.nn import FlopCountAnalysis, parameter_count_table
        flops = FlopCountAnalysis(model, x)
        print("FLOPs: %.4f G" % (flops.total()/1e9))

        total_paramters = 0
        for parameter in model.parameters():
            i = len(parameter.size())
            p = 1
            for j in range(i):
                p *= parameter.size(j)
            total_paramters += p
        print("Params: %.4f M" % (total_paramters / 1e6)) 