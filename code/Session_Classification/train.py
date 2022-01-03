# %%
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
import torch.nn as nn
import torch
from torchmetrics import Accuracy
from my_dataloader import *
from torch.utils.data import DataLoader
from cnn_model import *
# %%
class Sequence_Modeling(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        # self.rnn = nn.GRU(input_size=1, hidden_size=500, num_layers=2, batch_first=True, bidirectional=True, dropout=0.6)
        # self.linear_0 = nn.Linear(1000, 100)
        # self.linear_1 = nn.Linear(100, 2)
        self.loss = nn.NLLLoss()
        self.softmax = nn.LogSoftmax(dim=0)
        self.accuracy = Accuracy()
        self.model = CNN_FC_layer(output_class=2)
        
    def forward(self, x):
        # output, h1 = self.rnn(x)
        # h2 = torch.relu(self.linear_0(output[:, -1, :]))
        # logits = self.softmax(torch.relu(self.linear_1(h2)))
        logits = self.model(x)
        logits = self.softmax(logits) 
        return logits
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = 1e-3)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        
        J = self.loss(logits, y)
        
        acc =self.accuracy(logits, y)
        
        log = {'train_acc' : acc}
        
        return {'loss':J, 'log':log, 'progress_bar':log}
    
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
        return {'val_loss': avg_val_loss, 'log':pbar, 'progress_bar': pbar}
    
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
SIGNAL_DATAPATH = '../../data/RRI'
MASTER_TABLE_DATAPATH = '../../data/master_table.csv'

df_orig = pd.read_csv(MASTER_TABLE_DATAPATH)

subject = list(set(df_orig['subject']))

from sklearn.model_selection import train_test_split
train_id, test_id = train_test_split(subject, test_size=0.2, random_state=1004)

train_data, test_data = df_orig.query("subject.isin(@train_id) & session == [1, 2]", engine='python').reset_index(drop=True), \
    df_orig.query("subject.isin(@test_id) & session == [1, 2]", engine='python').reset_index(drop=True)

train_dataset = CustomDataset(data_table=train_data, data_dir=SIGNAL_DATAPATH)
test_dataset = CustomDataset(data_table=test_data, data_dir=SIGNAL_DATAPATH)

trainset_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=padd_seq, num_workers=4)
testset_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=padd_seq, num_workers=4)
# %%
class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        # bar.set_description('running validation ...')
        return bar
bar = LitProgressBar()
# %%
from pytorch_lightning.loggers import TensorBoardLogger

model = Sequence_Modeling()

logger = TensorBoardLogger("tb_logs", name="my_model")

trainer = pl.Trainer(logger=logger,
                     max_epochs=100, 
                     gpus=1, 
                     gradient_clip_val=0.5, 
                     log_every_n_steps=2, 
                     accumulate_grad_batches=2,
                     callbacks=[bar])
# %%
trainer.fit(model, 
            train_dataloaders = trainset_loader, 
            val_dataloaders = testset_loader)
# %%
