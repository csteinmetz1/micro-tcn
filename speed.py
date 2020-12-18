import os
import sys
import time
import torch
import numpy as np
import torchsummary
import pytorch_lightning as pl
from argparse import ArgumentParser

# import our models
from microtcn.tcn import TCNModel
from microtcn.lstm import LSTMModel


parser = ArgumentParser()
parser = TCNModel.add_model_specific_args(parser)   # add model specific args
parser = pl.Trainer.add_argparse_args(parser)       # add all the available trainer options to argparse
args = parser.parse_args()                          # parse them args

pl.seed_everything(42) # set the seed

dict_args = vars(args)
dict_args["nparams"] = 2
dict_args["kernel_size"] = 15
dict_args["channel_width"] = 32
dict_args["grouped"] = False
dict_args["dilation_growth"] = 2

model_type = "TCN"
gpu = False
sr = 44100
N = 44100
duration = N/sr # seconds 
n_iters = 256
timings = []

if model_type == "TCN": 
    model = TCNModel(**dict_args) # create the model with args
    rf = model.compute_receptive_field()
    input = (torch.rand(1,1,N+rf) * 2) - 1
else:
    model = torch.jit.script(LSTMModel(**dict_args)) # create the model with args
    input = (torch.rand(1,1,N) * 2) - 1

# count number of parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"{model_type} has {num_params} parameters")

if dict_args["nparams"] > 0:
    params = torch.rand(1,1,2)
else:
    params = None

#torchsummary.summary(model, [(1,N), (1,2)], device="cpu")

if gpu:
    model.cuda()
    input = input.to("cuda:0")
    params = params.to("cuda:0")

model.eval()
with torch.no_grad():
    for n in range(n_iters):
        tic = time.perf_counter()
        output = model(input, params)
        toc = time.perf_counter()
        timings.append(toc-tic)
        sys.stdout.write(f"{n+1:3d}/{n_iters:3d}\r")
        sys.stdout.flush()

print(output.shape)
mean_time_s = np.mean(timings)
mean_time_ms = mean_time_s * 1e3
sec_sec = (1/duration) * mean_time_s
print(f"Avg. time: {mean_time_ms:0.1f} ms  | sec/sec {sec_sec:0.3f} |  RTF: {duration/mean_time_s:0.2f}x")


