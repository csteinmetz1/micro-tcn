import torch
import pytorch_lightning as pl

class LSTMModel(pl.LightningModule):
    def __init__(self, 
                 nparams,
                 ninputs=1,
                 noutputs=1,
                 hidden_size=32,
                 num_layers=1,
                 num_examples=4,
                 save_dir=None,
                 **kwargs):
        super(LSTMModel, self).__init__()
        self.save_hyperparameters()

        input_size = ninputs + nparams
        self.lstm = torch.nn.LSTM(input_size,
                                   self.hparams.hidden_size,
                                   self.hparams.num_layers,
                                   batch_first=False,
                                   bidirectional=False)
        
        self.linear = torch.nn.Linear(self.hparams.hidden_size, 
                                      self.hparams.noutputs)

    def forward(self, x, p):

        bs = x.size(0) # batch size
        s = x.size(-1) # samples

        x = x.permute(2,0,1)

        if p is not None:
            p = p.repeat(s,bs,1) # expand to every time step
            x = torch.cat((x, p), dim=-1) # append to input along feature dim

        out, _ = self.lstm(x)
        out = self.linear(out)

        return out
