import os
import glob
import torch
from argparse import ArgumentParser

from microtcn.tcn import TCNModel
from microtcn.lstm import LSTMModel

def load_model(model_dir, gpu=False):

    checkpoint_path = glob.glob(os.path.join(model_dir,
                                            "lightning_logs",
                                            "version_0",
                                            "checkpoints",
                                            "*"))[0]

    hparams_file = os.path.join(model_dir, "hparams.yaml")
    batch_size = int(os.path.basename(model_id).split('-')[-1][2:])
    model_type = os.path.basename(model_id).split('-')[1]
    epoch = int(os.path.basename(checkpoint_path).split('-')[0].split('=')[-1])

    map_location = "cuda:0" if gpu else "cpu"

    if model_type == "LSTM":
        model = LSTMModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=map_location
        )

    else:
        model = TCNModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=map_location
        )

    return model

if __name__ == '__main__':

    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument('--model_dir', type=str, default='./lightning_logs/bulk')
    parser.add_argument('--save_dir', type=str, default='./models')

    # parse them args
    args = parser.parse_args()

    models = sorted(glob.glob(os.path.join(args.model_dir, "*")))

    for idx, model_dir in enumerate(models):

        model_id = os.path.basename(model_dir)
        print(model_id)
        model = load_model(model_dir)
        script = model.to_torchscript()
        #model = torch.jit.script(model) # create the model with args

        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)

        torch.jit.save(script, os.path.join(args.save_dir, f"traced_{model_id}.pt"))