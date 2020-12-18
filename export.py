import os
import torch

from microtcn.tcn_bare import TCNModel
from microtcn.lstm import LSTMModel

args = {
    "nparams" : 2
}

model = torch.jit.script(LSTMModel(**args)) # create the model with args

if not os.path.isdir("models"):
    os.makedirs("models")
model.save("models/traced_lstm_model.pt")