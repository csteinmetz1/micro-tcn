import os
import glob
import torch
import torchsummary
import pytorch_lightning as pl
from argparse import ArgumentParser

from microtcn.tcn import TCNModel
from microtcn.data import SignalTrainLA2ADataset

parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument('--root_dir', type=str, default='./data')
parser.add_argument('--preload', type=bool, default=False)
parser.add_argument('--sample_rate', type=int, default=44100)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--train_subset', type=str, default='train')
parser.add_argument('--val_subset', type=str, default='val')
parser.add_argument('--train_length', type=int, default=16384)
parser.add_argument('--eval_length', type=int, default=131072)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=16)

# add model specific args
parser = TCNModel.add_model_specific_args(parser)

# add all the available trainer options to argparse
parser = pl.Trainer.add_argparse_args(parser)

# parse them args
args = parser.parse_args()

# setup the dataloaders
train_dataset = SignalTrainLA2ADataset(args.root_dir, 
                                subset=args.train_subset,
                                half=True if args.precision == 16 else False,
                                preload=args.preload,
                                length=args.train_length)

train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                               shuffle=args.shuffle,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers)

val_dataset = SignalTrainLA2ADataset(args.root_dir, 
                                preload=args.preload,
                                half=True if args.precision == 16 else False,
                                subset=args.val_subset,
                                length=args.eval_length)

val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                             shuffle=False,
                                             batch_size=2,
                                             num_workers=args.num_workers)

# init the trainer and model 
trainer = pl.Trainer.from_argparse_args(args)
print(trainer.default_root_dir)

# set the seed
pl.seed_everything(42)

# create the model with args
dict_args = vars(args)
dict_args["nparams"] = 2
model = TCNModel(**dict_args)

torchsummary.summary(model.cuda(), [(1,args.train_length), (1,2)], device="cuda")

# train!
trainer.fit(model, train_dataloader, val_dataloader)
