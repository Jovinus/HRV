# %%
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torchmetrics import Accuracy
from my_dataloader import *
from torch.utils.data import DataLoader
# %%
class Sequence_Modeling(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_size=1, hidden_size=1000, num_layers=4, batch_first=True, bidirectional=True, dropout=0.6)
        self.linear_0 = nn.Linear(2000, 100)
        self.linear_1 = nn.Linear(100, 3)
        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.accuracy = Accuracy()
        
    def forward(self, x):
        output, h1 = self.rnn(x)
        h2 = torch.relu(self.linear_0(output[:, -1, :].view(-1, 2000)))
        logits = self.softmax(self.linear_1(h2))
        return logits
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = 1e-3)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        
        J = self.loss(logits, y)
        
        acc =self.accuracy(logits, y)
        
        pbar = {'train_acc' : acc}
        
        return {'loss':J, 'progress_bar':pbar}
    
    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        results['progress_bar']['val_acc'] = results['progress_bar']['train_acc']
        del results['progress_bar']['train_acc']
        return results
    
    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()
        avg_val_acc = torch.tensor([x['progress_bar']['val_acc'] for x in val_step_outputs]).mean()
        print(avg_val_acc)
        pbar = {'avg_val_acc': avg_val_acc}
        return {'val_loss': avg_val_loss, 'progress_bar': pbar}
    
    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        acc = self.accuracy(logits, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, 'test')
# %%

dataset = CustomDataset("../data/RRI")
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=padd_seq, num_workers=4)
# %%
model = Sequence_Modeling()
trainer = pl.Trainer(progress_bar_refresh_rate = 2, max_epochs=10, gpus=1, gradient_clip_val=0.5, log_every_n_steps=2)
# %%
trainer.fit(model, train_loader)