import os
import glob
import torch
import torchsummary
import pytorch_lightning as pl
from argparse import ArgumentParser
from itertools import product

from microtcn.tcn import TCNModel
from microtcn.lstm import LSTMModel
from microtcn.data import SignalTrainLA2ADataset

model_configs = [
    {"name" : "TCN-100",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 5
    },
    {"name" : "TCN-300",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 13
    },
    {"name" : "TCN-1000",
     "model_type" : "tcn",
     "nblocks" : 5,
     "dilation_growth" : 10,
     "kernel_size" : 5
    },
    {"name" : "LSTM-32",
     "model_type" : "lstm",
     "num_layers" : 1,
     "hidden_size" : 32,
    }
]

data_configs = [
    {"train_fraction" : 1.00},  # 100.0%
    {"train_fraction" : 0.10},  #  10.0%
    {"train_fraction" : 0.01},  #   1.0%
    {"train_fraction" : 0.001}, #   0.1%
]

configs = product(model_configs, data_configs)
n_configs = len(data_configs) * len(model_configs)

for idx, (t_conf, d_conf) in enumerate(configs):

    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument('--model_type', type=str, default='tcn', help='tcn or lstm')
    parser.add_argument('--root_dir', type=str, default='./data')
    parser.add_argument('--preload', action="store_true")
    parser.add_argument('--sample_rate', type=int, default=44100)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--train_subset', type=str, default='train')
    parser.add_argument('--val_subset', type=str, default='val')
    parser.add_argument('--train_length', type=int, default=65536)
    parser.add_argument('--train_fraction', type=float, default=1.0)
    parser.add_argument('--eval_length', type=int, default=131072)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=16)

    # add all the available trainer options to argparse
    parser = pl.Trainer.add_argparse_args(parser)

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    print(f"* Training config {idx+1}/{n_configs}")
    print(t_conf, d_conf)

    # set the seed
    pl.seed_everything(42)
  
    # let the model add what it wants
    if temp_args.model_type == 'tcn':
        parser = TCNModel.add_model_specific_args(parser)
    elif temp_args.model_type == 'lstm':
        parser = LSTMModel.add_model_specific_args(parser)

    # parse them args
    args = parser.parse_args()

    # init the trainer and model 
    if t_conf["model_type"] == 'tcn':
        specifier =  f"{idx+1}-{t_conf['model_type']}"
        specifier += f"__{t_conf['nblocks']}-{t_conf['dilation_growth']}-{t_conf['kernel_size']}"
        specifier += f"__fraction-{d_conf['train_fraction']}"
    elif t_conf["model_type"] == 'lstm':
        specifier =  f"{idx+1}-{t_conf['model_type']}"
        specifier += f"__{t_conf['num_layers']}-{t_conf['hidden_size']}"
        specifier += f"__fraction-{d_conf['train_fraction']}"

    args.default_root_dir = os.path.join("lightning_logs", "bulk", specifier)
    print(args.default_root_dir)
    trainer = pl.Trainer.from_argparse_args(args)

    # setup the dataloaders
    train_dataset = SignalTrainLA2ADataset(args.root_dir, 
                                    subset=args.train_subset,
                                    fraction=d_conf["train_fraction"],
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

    # create the model with args
    dict_args = vars(args)
    dict_args["nparams"] = 2

    if t_conf["model_type"] == 'tcn':
        dict_args["nblocks"] = t_conf["nblocks"]
        dict_args["dilation_growth"] = t_conf["dilation_growth"]
        dict_args["kernel_size"] = t_conf["kernel_size"]
        model = TCNModel(**dict_args)
    elif t_conf["model_type"] == 'lstm':
        dict_args["num_layers"] = t_conf["num_layers"]
        dict_args["hidden_size"] = t_conf["hidden_size"]
        model = LSTMModel(**dict_args)

    # train!
    trainer.fit(model, train_dataloader, val_dataloader)
