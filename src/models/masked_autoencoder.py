# %% ##################################### 
 # Descripttion: 
 # version: 
 # Author: Yuanjie Gu @ Fudan
 # Date: 2024-06-05
 # LastEditors: Yuanjie Gu
 # LastEditTime: 2025-05-09
# %% #####################################
# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .swin_transformer import SwinTransformer
from .vision_transformer import VisionTransformer




class SwinTransformerForMaskedAutoEncoder(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, mask):
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape
        mask_tokens = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        
        x = x * (1. - w) + mask_tokens * w

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        x = x.transpose(1, 2)
        B, C, L = x.shape
        H = W = int(L ** 0.5)
        x = x.reshape(B, C, H, W)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}


class VisionTransformerForMaskedAutoEncoder(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self._trunc_normal_(self.mask_token, std=.02)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def interpolate_pos_encoding(self,pos_embed, x, h, w): # interplate for dynamic inputs
        npatch = x.shape[1]
        N = pos_embed.shape[1]
        if npatch == N and w == h:
            return pos_embed
        patch_pos_embed = pos_embed
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)),
                                    dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(
            w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(self, x, mask):
        _, _, H_in, W_in = x.shape
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            self.pos_embed = self.interpolate_pos_encoding(self.pos_embed,x,H_in,W_in)
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x = self.norm(x)

        x = x[:, 1:]
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x

class VisionTransformerForParameterizedNoise(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.num_classes == 7

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self._trunc_normal_(self.mask_token, std=.02)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def interpolate_pos_encoding(self,pos_embed, x, h, w): # interplate for dynamic inputs
        npatch = x.shape[1]
        N = pos_embed.shape[1]
        if npatch == N and w == h:
            return pos_embed
        patch_pos_embed = pos_embed
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)),
                                    dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(
            w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

class ConvBlock(nn.Module):
    def __init__(self, inChannels, growRate):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(inChannels, growRate, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv(x))
        return torch.cat((x, out), 1)

class ResidualDenseBlock(nn.Module):
    def __init__(self, nFeatures=64, growRate=32, nConvLayers=6):
        super(ResidualDenseBlock, self).__init__()
               
        modules = []
        for i in range(nConvLayers):
            modules.append(ConvBlock(nFeatures + i * growRate, growRate))
        self.dense_layers = nn.Sequential(*modules)
        self.local_feature_fusion = nn.Conv2d(nFeatures + nConvLayers * growRate, nFeatures, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.local_feature_fusion(out)
        return out + x


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv(x))
        return torch.cat([x, out], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # Local Feature Fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, in_channels, kernel_size=1)

    def forward(self, x):
        out = self.layers(x)
        return x + self.lff(out)  # Local Residual Learning


class RDAdaptor(nn.Module): # thanks for https://github.com/yjn870/RDN-pytorch
    def __init__(self, num_channels, num_features, growth_rate, num_blocks, num_layers):
        super(RDAdaptor, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # Shallow Feature Extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        # Residual Dense Blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G0, self.G, self.C))

        # Global Feature Fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G0 * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=1)
        )

        # Output Layer
        self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # Global Residual Learning
        x = self.output(x)
        return x
    
class MaskedAutoEncoder(nn.Module):
    def __init__(self, encoder, encoder_stride, mode):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride
        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size
        self.mode = mode

        if self.mode == 'stage_1_1':
            self.decoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.encoder.num_features,
                    out_channels=self.encoder_stride ** 2 * self.in_chans, kernel_size=1),
                nn.PixelShuffle(self.encoder_stride),
            )
        elif self.mode == 'stage_2':
            self.decoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.encoder.num_features,
                    out_channels=self.encoder_stride ** 2 * self.in_chans, kernel_size=1),
                nn.PixelShuffle(self.encoder_stride),
            )
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
        elif self.mode == 'stage_3':
            self.decoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.encoder.num_features,
                    out_channels=self.encoder_stride ** 2 * self.in_chans, kernel_size=1),
                nn.PixelShuffle(self.encoder_stride),
            )
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
        else: assert NotImplementedError(f"Unkown mode{self.mode}, must among in 'stage_1_1', 'stage_2' and 'stage_3'")


    def forward(self, x, mask):
        z = self.encoder(x, mask)
        x_rec = self.decoder(z)
        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
        return x_rec, mask
        

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}

class Adaptor(nn.Module):
    def __init__(self,in_chans, decoder_nfeatures, stage):
        super().__init__()
        self.adaptor = nn.Sequential(
                nn.Conv2d(in_chans, decoder_nfeatures // 2,kernel_size=3,padding=1),
                nn.Conv2d(decoder_nfeatures // 2, decoder_nfeatures,kernel_size=3,padding=1),
                ResidualDenseBlock(nFeatures=decoder_nfeatures, growRate=32, nConvLayers=6),
                ResidualDenseBlock(nFeatures=decoder_nfeatures, growRate=32, nConvLayers=6),
                ResidualDenseBlock(nFeatures=decoder_nfeatures, growRate=32, nConvLayers=6),
                ResidualDenseBlock(nFeatures=decoder_nfeatures, growRate=32, nConvLayers=6),
                nn.Conv2d(decoder_nfeatures, decoder_nfeatures // 2,kernel_size=3,padding=1),
                nn.Conv2d(decoder_nfeatures//2, in_chans,kernel_size=3,padding=1)
            )
        if stage == 'stage_3':
            for param in self.adaptor.parameters():
                param.requires_grad = False
    def forward(self,x):
        return self.adaptor(x) + x


class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(PatchGANDiscriminator, self).__init__()
        
        # Initial convolution layer
        model = [nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]
        
        # Downsampling layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            model += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=4, stride=2, padding=1, bias=False),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        # Final layer without downsampling
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        model += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=4, stride=1, padding=1, bias=False),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Output layer
        model += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

def build_discriminator(input_channels=3, ndf=64, n_layers=3):
    discriminator = PatchGANDiscriminator(input_channels=input_channels, ndf=ndf, n_layers=n_layers)
    return discriminator
def build_adaptor(in_chans=3,decoder_nfeatures=64, stage = 'Stage_2'):
    adaptor = Adaptor(in_chans=in_chans,decoder_nfeatures=decoder_nfeatures, stage = stage)
    return adaptor

'''
def build_adaptor(in_chans=3,decoder_nfeatures=64, growth_rate=32, num_blocks=4, num_layers=6):
    adaptor = RDAdaptor(num_channels=in_chans, num_features=decoder_nfeatures, growth_rate=growth_rate, num_blocks=num_blocks, num_layers=num_layers)
    return adaptor
'''

def build_masked_autoencoder(model_type = 'swin',img_size = 192, patch_size = 4, in_chans = 3, mode = 'pretrain'):
    if model_type == 'swin':
        encoder = SwinTransformerForMaskedAutoEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,
            embed_dim=128,
            depths=[2,2,18,2],
            num_heads=[4,8,16,32],
            window_size=6,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            drop_path_rate=0.0,
            ape=False,
            patch_norm=True,
            use_checkpoint=False)
        encoder_stride = 32
    elif model_type == 'vit':
        encoder = VisionTransformerForMaskedAutoEncoder(
            img_size,
            patch_size,
            in_chans,
            num_classes=0,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=0.1,
            use_abs_pos_emb=False,
            use_rel_pos_bias=False,
            use_shared_rel_pos_bias=True,
            use_mean_pooling=False)
        encoder_stride = 16
    else:
        raise NotImplementedError(f"Unknown encoder model: {model_type}")
    model = MaskedAutoEncoder(encoder=encoder, encoder_stride=encoder_stride, mode=mode)
    return model

def build_noiser(model_type='vit', img_size=192, patch_size=16, in_chans=3, return_attn=False):
    if model_type == 'swin':
        model = SwinTransformer(
            img_size=192,
            patch_size=4,
            in_chans=3,
            num_classes=6,
            embed_dim=128,
            depths=[2,2,18,2],
            num_heads=[4,8,16,32],
            window_size=6,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            drop_path_rate=0.0,
            ape=False,
            patch_norm=True,
            use_checkpoint=False
        )
    elif model_type == 'vit':
        model = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=6,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=0.1,
            use_abs_pos_emb=True,
            use_rel_pos_bias=False,
            use_shared_rel_pos_bias=False,
            use_mean_pooling=False
        )
    # Attach return_attn as an attribute for downstream usage
    model.return_attn = return_attn
    return model


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, use_dropout=False):
        super().__init__()
        self.down = down
        if down:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        if self.down:
            x = self.block(x)
        else:
            # 先上采样再卷积
            x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = self.block(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x

class DDPM_UNet(nn.Module):
    """
    简单的DDPM用Unet结构，适用于去噪扩散模型
    """
    def __init__(self, in_ch=3, out_ch=3, base_ch=64):
        super().__init__()
        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc2 = UNetBlock(base_ch, base_ch*2, down=True)
        self.enc3 = UNetBlock(base_ch*2, base_ch*4, down=True)
        self.enc4 = UNetBlock(base_ch*4, base_ch*8, down=True)
        # 中间
        self.middle = nn.Sequential(
            nn.Conv2d(base_ch*8, base_ch*8, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        # 解码器
        self.dec4 = UNetBlock(base_ch*8, base_ch*4, down=False)
        self.dec3 = UNetBlock(base_ch*8, base_ch*2, down=False)
        self.dec2 = UNetBlock(base_ch*4, base_ch, down=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch*2, out_ch, 3, padding=1)
        )
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        m = self.middle(e4)
        d4 = self.dec4(m)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        return d1

def build_ddpm_unet(in_ch=3, out_ch=3, base_ch=64):
    return DDPM_UNet(in_ch=in_ch, out_ch=out_ch, base_ch=base_ch)

if __name__ == "__main__":
    input = torch.randn(2,3,192,192)
    #model = build_simmim(model_type='swin',mode='finetune')
    model = build_adaptor()
    output = model(input)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6}M")
    print(f"Trainable parameters: {trainable_params/1e6}M")
    print(output.shape)

    # 测试DDPM_UNet模型
    input = torch.randn(2, 3, 128, 128)
    model = build_ddpm_unet(in_ch=3, out_ch=3, base_ch=64)
    output = model(input)
    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6}M")
    print(f"Trainable parameters: {trainable_params/1e6}M")