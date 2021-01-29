import os
import sys
import glob
import json
import torch
import pickle
import torchaudio
import numpy as np
import torchsummary
from thop import profile
import pyloudnorm as pyln
import pytorch_lightning as pl
from argparse import ArgumentParser

import auraloss

from microtcn.tcn import TCNModel
from microtcn.lstm import LSTMModel
from microtcn.data import SignalTrainLA2ADataset
from microtcn.utils import center_crop, causal_crop

parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument('--root_dir', type=str, default='./data')
parser.add_argument('--model_dir', type=str, default='./lightning_logs/bulk')
parser.add_argument('--save_dir', type=str, default=None)
parser.add_argument('--preload', action="store_true", default=False)
parser.add_argument('--half', action="store_true", default=False)
parser.add_argument('--fast', action="store_true", default=False) # skip LSTM
parser.add_argument('--sample_rate', type=int, default=44100)
parser.add_argument('--eval_subset', type=str, default='val')
parser.add_argument('--eval_length', type=int, default=8388608)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=32)

# parse them args
args = parser.parse_args()

# set the seed
pl.seed_everything(42)

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

overall_results = {}

if args.save_dir is not None:
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

# set up loss functions for evaluation
l1   = torch.nn.L1Loss()
stft = auraloss.freq.STFTLoss()
meter = pyln.Meter(44100)

models = sorted(glob.glob(os.path.join(args.model_dir, "*")))

for idx, model_dir in enumerate(models):

    results = {}

    checkpoint_path = glob.glob(os.path.join(model_dir,
                                             "lightning_logs",
                                             "version_0",
                                             "checkpoints",
                                             "*"))[0]
    hparams_file = os.path.join(model_dir, "hparams.yaml")

    model_id = os.path.basename(model_dir)
    batch_size = int(os.path.basename(model_dir).split('-')[-1][2:])
    model_type = os.path.basename(model_dir).split('-')[1]
    epoch = int(os.path.basename(checkpoint_path).split('-')[0].split('=')[-1])

    if model_type == "LSTM":
        if args.fast: continue
        model = LSTMModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location="cuda:0"
        )

    else:
        model = TCNModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location="cuda:0"
        )

    i = torch.rand(1,1,65536)
    p = torch.rand(1,1,2)
    #macs, params = profile(model, inputs=(i, p))

    print(f" {idx+1}/{len(models)} : epoch: {epoch} {os.path.basename(model_dir)} batch size {batch_size}")
    #print(   f"MACs: {macs/10**9:0.2f} G     Params: {params/1e3:0.2f} k")

    model.cuda()
    model.eval()

    if args.half:
        model.half()

    # set the seed
    pl.seed_everything(42)

    for bidx, batch in enumerate(test_dataloader):

        sys.stdout.write(f" Evaluating {bidx}/{len(test_dataloader)}...\r")
        sys.stdout.flush()

        input, target, params = batch

        # move to gpu
        input = input.to("cuda:0")
        target = target.to("cuda:0")
        params = params.to("cuda:0")

        with torch.no_grad(), torch.cuda.amp.autocast():
            output = model(input, params)

            # crop the input and target signals
            if model.hparams.causal:
                input_crop = causal_crop(input, output.shape[-1])
                target_crop = causal_crop(target, output.shape[-1])
            else:
                input_crop = center_crop(input, output.shape[-1])
                target_crop = center_crop(target, output.shape[-1])


        for idx, (i, o, t, p) in enumerate(zip(
                                            torch.split(input_crop, 1, dim=0),
                                            torch.split(output, 1, dim=0),
                                            torch.split(target_crop, 1, dim=0),
                                            torch.split(params, 1, dim=0))):

            l1_loss = l1(o, t).cpu().numpy()
            stft_loss = stft(o, t).cpu().numpy()
            aggregate_loss = l1_loss + stft_loss 

            target_lufs = meter.integrated_loudness(t.squeeze().cpu().numpy())
            output_lufs = meter.integrated_loudness(o.squeeze().cpu().numpy())
            l1_lufs = np.abs(output_lufs - target_lufs)

            l1i_loss = (l1(i, t) - l1(o, t)).cpu().numpy()
            stfti_loss = (stft(i, t) - stft(o, t)).cpu().numpy()

            params = p.squeeze().cpu().numpy()
            params_key = f"{params[0]:1.0f}-{params[1]*100:03.0f}"

            if args.save_dir is not None:
                ofile = os.path.join(args.save_dir, f"{params_key}-{bidx}-output--{model_id}.wav")
                ifile = os.path.join(args.save_dir, f"{params_key}-{bidx}-input.wav")
                tfile = os.path.join(args.save_dir, f"{params_key}-{bidx}-target.wav")

                torchaudio.save(ofile, o.view(1,-1).cpu().float(), 44100)
                if not os.path.isfile(ifile):
                    torchaudio.save(ifile, i.view(1,-1).cpu().float(), 44100)
                if not os.path.isfile(tfile):
                    torchaudio.save(tfile, t.view(1,-1).cpu().float(), 44100)

            if params_key not in list(results.keys()):
                results[params_key] = {
                    "L1" : [l1_loss],
                    "L1i" : [l1i_loss],
                    "STFT" : [stft_loss],
                    "STFTi" : [stfti_loss],
                    "LUFS" : [l1_lufs],
                    "Agg" : [aggregate_loss]
                }
            else:
                results[params_key]["L1"].append(l1_loss)
                results[params_key]["L1i"].append(l1i_loss)
                results[params_key]["STFT"].append(stft_loss)
                results[params_key]["STFTi"].append(stfti_loss)
                results[params_key]["LUFS"].append(l1_lufs)
                results[params_key]["Agg"].append(aggregate_loss)

    # store in dict
    l1_scores = []
    lufs_scores = []
    stft_scores = []
    agg_scores = []
    print("-" * 64)
    print("Config      L1         STFT      LUFS")
    print("-" * 64)
    for key, val in results.items():
        print(f"{key}    {np.mean(val['L1']):0.2e}    {np.mean(val['STFT']):0.3f}       {np.mean(val['LUFS']):0.3f}")

        l1_scores += val["L1"]
        stft_scores += val["STFT"]
        lufs_scores += val["LUFS"]
        agg_scores += val["Agg"]

    print("-" * 64)
    print(f"Mean     {np.mean(l1_scores):0.2e}    {np.mean(stft_scores):0.3f}      {np.mean(lufs_scores):0.3f}")
    print()
    overall_results[model_id] = results

pickle.dump(overall_results, open(f"test_results_{args.eval_subset}.p", "wb" ))

# we can make some kind of scatter plot to visualize this