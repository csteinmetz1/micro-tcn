import os
import torch
import torchaudio
import numpy as np
import torchsummary
import pytorch_lightning as pl
from argparse import ArgumentParser

from microtcn.base import Base
from microtcn.utils import center_crop, causal_crop

class FiLM(torch.nn.Module):
    def __init__(self, 
                 num_features, 
                 cond_dim):
        super(FiLM, self).__init__()
        self.num_features = num_features
        self.bn = torch.nn.BatchNorm1d(num_features, affine=False)
        self.adaptor = torch.nn.Linear(cond_dim, num_features * 2)

    def forward(self, x, cond):

        cond = self.adaptor(cond)
        g, b = torch.chunk(cond, 2, dim=-1)
        g = g.permute(0,2,1)
        b = b.permute(0,2,1)

        x = self.bn(x)      # apply BatchNorm without affine
        x = (x * g) + b     # then apply conditional affine

        return x

class TCNBlock(torch.nn.Module):
    def __init__(self, 
                in_ch, 
                out_ch, 
                kernel_size=3, 
                padding="same", 
                dilation=1, 
                grouped=False, 
                causal=False,
                conditional=False, 
                **kwargs):
        super(TCNBlock, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.grouped = grouped
        self.causal = causal
        self.conditional = conditional

        groups = out_ch if grouped and (in_ch % out_ch == 0) else 1

        if padding == "same":
            pad_value = (kernel_size - 1) + ((kernel_size - 1) * (dilation-1))
        elif padding in ["none", "valid"]:
            pad_value = 0

        self.conv1 = torch.nn.Conv1d(in_ch, 
                                     out_ch, 
                                     kernel_size=kernel_size, 
                                     padding=0, # testing a change in padding was pad_value//2
                                     dilation=dilation,
                                     groups=groups,
                                     bias=False)
        if grouped:
            self.conv1b = torch.nn.Conv1d(out_ch, out_ch, kernel_size=1)

        if conditional:
            self.film = FiLM(out_ch, 32)
        else:
            self.bn = torch.nn.BatchNorm1d(out_ch)

        self.relu = torch.nn.PReLU(out_ch)
        self.res = torch.nn.Conv1d(in_ch, 
                                   out_ch, 
                                   kernel_size=1,
                                   groups=in_ch,
                                   bias=False)

    def forward(self, x, p):
        x_in = x

        x = self.conv1(x)
        #if self.grouped: # apply pointwise conv
        #    x = self.conv1b(x)
        #if p is not None:   
        x = self.film(x, p) # apply FiLM conditioning
        #else:
        #    x = self.bn(x)
        x = self.relu(x)

        x_res = self.res(x_in)
        if self.causal:
            x = x + causal_crop(x_res, x.shape[-1])
        else:
            x = x + center_crop(x_res, x.shape[-1])

        return x

class TCNModel(Base):
    """ Temporal convolutional network with conditioning module.

        Args:
            nparams (int): Number of conditioning parameters.
            ninputs (int): Number of input channels (mono = 1, stereo 2). Default: 1
            noutputs (int): Number of output channels (mono = 1, stereo 2). Default: 1
            nblocks (int): Number of total TCN blocks. Default: 10
            kernel_size (int): Width of the convolutional kernels. Default: 3
            dialation_growth (int): Compute the dilation factor at each block as dilation_growth ** (n % stack_size). Default: 1
            channel_growth (int): Compute the output channels at each black as in_ch * channel_growth. Default: 2
            channel_width (int): When channel_growth = 1 all blocks use convolutions with this many channels. Default: 64
            stack_size (int): Number of blocks that constitute a single stack of blocks. Default: 10
            grouped (bool): Use grouped convolutions to reduce the total number of parameters. Default: False
            causal (bool): Causal TCN configuration does not consider future input values. Default: False
            skip_connections (bool): Skip connections from each block to the output. Default: False
            num_examples (int): Number of evaluation audio examples to log after each epochs. Default: 4
        """
    def __init__(self, 
                 nparams,
                 ninputs=1,
                 noutputs=1,
                 nblocks=10, 
                 kernel_size=3, 
                 dilation_growth=1, 
                 channel_growth=1, 
                 channel_width=32, 
                 stack_size=10,
                 grouped=False,
                 causal=False,
                 skip_connections=False,
                 num_examples=4,
                 save_dir=None,
                 **kwargs):
        super(TCNModel, self).__init__()
        self.save_hyperparameters()

        if self.hparams.nparams > 0:
            self.gen = torch.nn.Sequential(
                torch.nn.Linear(nparams, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 32),
                torch.nn.ReLU()
            )

        self.blocks = torch.nn.ModuleList()
        for n in range(nblocks):
            in_ch = out_ch if n > 0 else ninputs
            
            if self.hparams.channel_growth > 1:
                out_ch = in_ch * self.hparams.channel_growth 
            else:
                out_ch = self.hparams.channel_width

            dilation = self.hparams.dilation_growth ** (n % self.hparams.stack_size)
            self.blocks.append(TCNBlock(in_ch, 
                                        out_ch, 
                                        kernel_size=self.hparams.kernel_size, 
                                        dilation=dilation,
                                        padding="same" if self.hparams.causal else "valid",
                                        causal=self.hparams.causal,
                                        grouped=self.hparams.grouped,
                                        conditional=True if self.hparams.nparams > 0 else False))

        self.output = torch.nn.Conv1d(out_ch, noutputs, kernel_size=1)

    def forward(self, x, p):
        # if parameters present, 
        # compute global conditioning
        #if p is not None:
        cond = self.gen(p)
        #else:
        #    cond = None

        # iterate over blocks passing conditioning
        for idx, block in enumerate(self.blocks):
            x = block(x, cond)
            #if self.hparams.skip_connections:
            #    if idx == 0:
            #        skips = x
            #    else:
            #        skips = center_crop(skips, x[-1]) + x
            #else:
            skips = 0

        out = torch.tanh(self.output(x + skips))

        return out

    def compute_receptive_field(self):
        """ Compute the receptive field in samples."""
        rf = self.hparams.kernel_size
        for n in range(1,self.hparams.nblocks):
            dilation = self.hparams.dilation_growth ** (n % self.hparams.stack_size)
            rf = rf + ((self.hparams.kernel_size-1) * dilation)
        return rf

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # --- model related ---
        parser.add_argument('--ninputs', type=int, default=1)
        parser.add_argument('--noutputs', type=int, default=1)
        parser.add_argument('--nblocks', type=int, default=4)
        parser.add_argument('--kernel_size', type=int, default=5)
        parser.add_argument('--dilation_growth', type=int, default=10)
        parser.add_argument('--channel_growth', type=int, default=1)
        parser.add_argument('--channel_width', type=int, default=32)
        parser.add_argument('--stack_size', type=int, default=10)
        parser.add_argument('--grouped', default=False, action='store_true')
        parser.add_argument('--causal', default=False, action="store_true")
        parser.add_argument('--skip_connections', default=False, action="store_true")

        return parser