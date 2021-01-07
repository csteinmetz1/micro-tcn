import os
import sys
import glob
import json
import torch
import numpy as np
import torchsummary
import pytorch_lightning as pl
from argparse import ArgumentParser

import auraloss

from microtcn.tcn import TCNModel
from microtcn.lstm import LSTMModel
from microtcn.data import SignalTrainLA2ADataset

def center_crop(x, shape):
    start = (x.shape[-1]-shape[-1])//2
    stop  = start + shape[-1]
    return x[...,start:stop]

parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument('--root_dir', type=str, default='./data')
parser.add_argument('--model_dir', type=str, default='./lightning_logs/bulk')
parser.add_argument('--preload', action="store_true", default=False)
parser.add_argument('--half', action="store_true", default=False)
parser.add_argument('--sample_rate', type=int, default=44100)
parser.add_argument('--eval_subset', type=str, default='val')
parser.add_argument('--eval_length', type=int, default=131072)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=32)

# parse them args
args = parser.parse_args()

# setup the dataloaders
test_dataset = SignalTrainLA2ADataset(args.root_dir, 
                                      subset=args.eval_subset,
                                      half=False,
                                      preload=args.preload,
                                      length=args.eval_length)

test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                               shuffle=False,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers)

results = {}

# set up loss functions for evaluation
l1   = torch.nn.L1Loss()
stft = auraloss.freq.STFTLoss()

models = sorted(glob.glob(os.path.join(args.model_dir, "*")))

for idx, model_dir in enumerate(models):

    checkpoint_path = glob.glob(os.path.join(model_dir,
                                             "lightning_logs",
                                             "version_0",
                                             "checkpoints",
                                             "*"))[0]
    hparams_file = os.path.join(model_dir, "hparams.yaml")

    model_type = os.path.basename(model_dir).split('-')[1]
    epoch = int(os.path.basename(checkpoint_path).split('-')[0].split('=')[-1])

    print(f" {idx+1}/{len(models)} : epoch: {epoch} {model_dir}")

    if model_type == "LSTM":
        model = LSTMModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location="cuda:0"
        )

    else:
        model = TCNModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location="cuda:0"
        )

    model.cuda()
    model.eval()

    if args.half:
        model.half()

    # set the seed
    pl.seed_everything(42)

    for bidx, batch in enumerate(test_dataloader):

        sys.stdout.write(f" {bidx}/{len(test_dataloader)}\r")
        sys.stdout.flush()

        input, target, params = batch

        # move to gpu
        input = input.to("cuda:0")
        target = target.to("cuda:0")
        params = params.to("cuda:0")

        with torch.no_grad(), torch.cuda.amp.autocast():
            output = model(input, params)
            target_crop = center_crop(target, output.shape)
            input_crop = center_crop(input, output.shape)

        for i, o, t, p in zip(torch.split(input_crop, 1, dim=0),
                           torch.split(output, 1, dim=0),
                           torch.split(target_crop, 1, dim=0),
                           torch.split(params, 1, dim=0)):

            l1_loss = l1(o, t).cpu().numpy()
            stft_loss = stft(o, t).cpu().numpy()
            aggregate_loss = l1_loss + stft_loss 

            l1i_loss = (l1(i, t) - l1(o, t)).cpu().numpy()
            stfti_loss = (stft(i, t) - stft(o, t)).cpu().numpy()

            params = p.squeeze().cpu().numpy()
            params_key = f"{params[0]:1.0f}-{params[1]*100:03.0f}"

            if params_key not in list(results.keys()):
                results[params_key] = {
                    "L1" : [l1_loss],
                    "L1i" : [l1i_loss],
                    "STFT" : [stft_loss],
                    "STFTi" : [stfti_loss],
                    "Agg" : [aggregate_loss]
                }
            else:
                results[params_key]["L1"].append(l1_loss)
                results[params_key]["L1i"].append(l1i_loss)
                results[params_key]["STFT"].append(stft_loss)
                results[params_key]["STFTi"].append(stfti_loss)
                results[params_key]["Agg"].append(aggregate_loss)

    # store in dict
    for key, val in results.items():
        print(f"{key}    L1: {np.mean(val['L1']):0.2e}    STFT: {np.mean(val['STFT']):0.3f}  STFTi: {np.mean(val['STFTi']):0.3f}   Agg: {np.mean(val['Agg']):0.3f}")

