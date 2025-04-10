import os
from contextlib import contextmanager
import torch
import numpy as np
from einops import rearrange
import torch.nn.functional as F
import torch.nn as nn
# import pytorch_lightning as pl
# from lvdm.modules.networks.ae_modules import Encoder, Decoder
# from lvdm.distributions import DiagonalGaussianDistribution
from utils.utils import instantiate_from_config
from lvdm.basics import (
    zero_module,
    conv_nd,
    linear,
    avg_pool_nd,
    normalization
)


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class VAEEncoderadaptor(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        assert self.num_resolutions == 1
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                # if curr_res in attn_resolutions:
                #     attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            # if i_level != self.num_resolutions-1:
            #     down.downsample = Downsample(block_in, resamp_with_conv)
            #     curr_res = curr_res // 2
            self.down.append(down)

        # # middle
        # self.mid = nn.Module()
        # self.mid.block_1 = ResnetBlock(in_channels=block_in,
        #                                out_channels=block_in,
        #                                temb_channels=self.temb_ch,
        #                                dropout=dropout)
        # self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        # self.mid.block_2 = ResnetBlock(in_channels=block_in,
        #                                out_channels=block_in,
        #                                temb_channels=self.temb_ch,
        #                                dropout=dropout)

        # # end
        # self.norm_out = Normalize(block_in)
        # self.conv_out = torch.nn.Conv2d(block_in,
        #                                 2*z_channels if double_z else z_channels,
        #                                 kernel_size=3,
        #                                 stride=1,
        #                                 padding=1)
        self.norm_out = Normalize(block_out)
        self.conv_out = zero_module(torch.nn.Conv2d(block_out,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1))
        

    def forward(self, x):
        # timestep embedding
        temb = None

        # print(f'encoder-input={x.shape}')
        # downsampling
        hs = [self.conv_in(x)]
        # print(f'encoder-conv in feat={hs[0].shape}')
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                # print(f'encoder-down feat={h.shape}')
                # if len(self.down[i_level].attn) > 0:
                #     h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            # if i_level != self.num_resolutions-1:
            #     # print(f'encoder-downsample (input)={hs[-1].shape}')
            #     hs.append(self.down[i_level].downsample(hs[-1]))
            #     # print(f'encoder-downsample (output)={hs[-1].shape}')

        # # middle
        # h = hs[-1]
        # h = self.mid.block_1(h, temb)
        # # print(f'encoder-mid1 feat={h.shape}')
        # h = self.mid.attn_1(h)
        # h = self.mid.block_2(h, temb)
        # # print(f'encoder-mid2 feat={h.shape}')

        # # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        # # print(f'end feat={h.shape}')
        return h + x



class VAEDecoderadaptor(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        assert self.num_resolutions == 1
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("AE working on z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # # z to block_in
        # self.conv_in = torch.nn.Conv2d(z_channels,
        #                                block_in,
        #                                kernel_size=3,
        #                                stride=1,
        #                                padding=1)

        # # middle
        # self.mid = nn.Module()
        # self.mid.block_1 = ResnetBlock(in_channels=block_in,
        #                                out_channels=block_in,
        #                                temb_channels=self.temb_ch,
        #                                dropout=dropout)
        # self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        # self.mid.block_2 = ResnetBlock(in_channels=block_in,
        #                                out_channels=block_in,
        #                                temb_channels=self.temb_ch,
        #                                dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                # if curr_res in attn_resolutions:
                #     attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            # if i_level != 0:
            #     up.upsample = Upsample(block_in, resamp_with_conv)
            #     curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        # #assert z.shape[1:] == self.z_shape[1:]
        # self.last_z_shape = z.shape

        # # print(f'decoder-input={z.shape}')
        # # timestep embedding
        temb = None
        h = z

        # # z to block_in
        # h = self.conv_in(z)
        # # print(f'decoder-conv in feat={h.shape}')

        # # middle
        # h = self.mid.block_1(h, temb)
        # h = self.mid.attn_1(h)
        # h = self.mid.block_2(h, temb)
        # # print(f'decoder-mid feat={h.shape}')

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
            #     if len(self.up[i_level].attn) > 0:
            #         h = self.up[i_level].attn[i_block](h)
            #     # print(f'decoder-up feat={h.shape}')
            # if i_level != 0:
            #     h = self.up[i_level].upsample(h)
            #     # print(f'decoder-upsample feat={h.shape}')

        # end
        # if self.give_pre_end:
        #     return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        # print(f'decoder-conv_out feat={h.shape}')
        if self.tanh_out:
            h = torch.tanh(h)
        return h
    
# class VAEadaptor(nn.Module):
#     def __init__(self,
#                  ddconfig,
#                  ):
#         super().__init__()
#         self.encoder_adaptor = VAEEncoderadaptor(**ddconfig)
#         self.decoder_adaptor = VAEDecoderadaptor(**ddconfig)