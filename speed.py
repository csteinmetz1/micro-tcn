import os
import sys
import time
import torch
import numpy as np
import torchsummary
import pandas as pd
from itertools import product
import pytorch_lightning as pl
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# import our models
from microtcn.tcn_bare import TCNModel
from microtcn.lstm import LSTMModel

def compute_receptive_field(nblocks, dilation_growth, kernel_size, stack_size=10):
    """ Compute the receptive field in samples."""
    rf = kernel_size
    for n in range(1,nblocks):
        dilation = dilation_growth ** (n % stack_size)
        rf = rf + ((kernel_size-1) * dilation)
    return rf

def run(nblocks, dilation_growth, kernel_size, target_rf, model_type="TCN", causal=False, N=44100, gpu=False):

    pl.seed_everything(42) # set the seed

    dict_args = {}
    dict_args["nparams"] = 2
    dict_args["nblocks"] = nblocks
    dict_args["kernel_size"] = kernel_size
    dict_args["channel_width"] = 32
    dict_args["hidden_size"] = 32
    dict_args["grouped"] = False
    dict_args["causal"] = causal
    dict_args["dilation_growth"] = dilation_growth

    sr = 44100
    #N = 44100
    duration = N/sr # seconds 
    n_iters = 10
    timings = []

    if model_type == "TCN": 
        rf = compute_receptive_field(nblocks, dilation_growth, kernel_size)
        samples = N+rf
        # don't construct model if rf is too large
        if target_rf != -1:
            if (rf/sr)*1e3 > target_rf * 2: 
                return rf, 0
            if (rf/sr)*1e3 < target_rf:
                return rf, 0

        model = TCNModel(**dict_args) # create the model with args
        input = (torch.rand(1,1,samples) * 2) - 1
    else:
        rf = 0
        model = LSTMModel(**dict_args) # create the model with args
        input = (torch.rand(1,1,N) * 2) - 1

    # count number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"{model_type} has {num_params} parameters with r.f. {(rf/sr)*1e3:0.1f} ms requiring input size {N+rf}")

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

    mean_time_s = np.mean(timings)
    mean_time_ms = mean_time_s * 1e3
    sec_sec = (1/duration) * mean_time_s
    rtf = duration/mean_time_s
    rf_ms = (rf/sr)*1e3
    print(f"Avg. time: {mean_time_ms:0.1f} ms  | sec/sec {sec_sec:0.3f} |  RTF: {rtf:0.2f}x")

    return rf_ms, rtf

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--rf", action="store_true")
    args = parser.parse_args()

    max_dilation = 128
    max_blocks = 5
    max_kernel = 33

    dilation_factors = np.arange(1,max_dilation+1)
    nblocks = np.arange(1,max_blocks+1)
    kernels = np.arange(3,max_kernel+1,step=2)

    candidates = []

    if args.full:
        for b, d, k in product(nblocks, dilation_factors, kernels):
            print(b, d, k)
            rf, rtf = run(b, d, k, args.rf, N=512)
            if rf > args.rf:
                candidates.append({
                    "kernel" : k,
                    "dilation": d,
                    "blocks" : b,
                    "rf" : rf,
                    "rtf" : rtf
                })
    else:
        frame_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
        causal      = [True, False]
        for c, N in product(causal, frame_sizes):
            model_id = ["TCN-370", "TCN-100", "TCN-300", "TCN-1000", "TCN-324", "LSTM-32"]
            model_type = ["TCN", "TCN", "TCN", "TCN", "TCN", "LSTM"]
            nblocks          = [ 3, 4,  4,  5, 10, 0]
            dilation_factors = [64,10, 10, 10,  2, 0]
            kernels          = [ 5, 5, 13,  5, 15, 0]
            for mid, m, b, d, k in zip(model_id, 
                                        model_type, 
                                        nblocks, 
                                        dilation_factors, 
                                        kernels):
                print(b, d, k)
                rf, rtf = run(b, d, k, -1, causal=c, N=N, model_type=m, gpu=args.gpu)
                if c:   mid += "-C"
                else:   mid += "-N"
                candidates.append({
                    "model_id": mid,
                    "causal" : c,
                    "kernel" : k,
                    "dilation": d,
                    "blocks" : b,
                    "rf" : rf,
                    "rtf" : rtf,
                    "N" : N
                })

    df = pd.DataFrame(candidates)
    print(df)
    if args.gpu:
        df.to_csv('speed_gpu.csv')
    else:
        df.to_csv('speed_cpu.csv')

    # find the optimal architecture
    #sorted_candidates = sorted(candidates, key = lambda x: x["rtf"], reverse=True)
    print("-"*50)
    print("     ID      RTF       RF      Blocks  Dilation   Kernel")
    print("-"*50)
    for n, c in enumerate(candidates[:11]):
        print(f"{n: 3d} {c['model_id']}  {c['rtf']: 2.2f}x  {c['rf']:0.1f} ms    {c['blocks']}        {c['dilation']}        {c['kernel']}")
    print("-"*50)

    if args.plot:
        fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(10,4))

        norm = colors.DivergingNorm(vmin=0, vcenter=1)
        img = ax[0].pcolormesh(dilation_factors, nblocks, rf_res.T / 1000)
        ax[0].set_xticks(nblocks)
        ax[0].set_yticks(dilation_factors)

        plt.colorbar(img, ax=ax[0])
        img = ax[1].pcolormesh(rtf_res.T,  cmap='RdBu', norm=norm)
        plt.colorbar(img, ax=ax[1])
        plt.show()