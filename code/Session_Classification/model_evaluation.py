# %%
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn as nn
import torch
from torchmetrics import Accuracy
from my_dataloader import *
from torch.utils.data import DataLoader
from residual_cnn_1d import *
from sklearn.model_selection import train_test_split, RepeatedKFold
# %%
class Stress_Classification(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.NLLLoss()
        self.softmax = nn.LogSoftmax(dim=0)
        self.accuracy = Accuracy()
        self.model = Residual_CNN_Model(output_class=2)
        
    def forward(self, x):
        logits = self.model(x)
        logits = self.softmax(logits) 
        return logits
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = 1e-3)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        acc = self.accuracy(logits, y)
        
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        val_acc = self.accuracy(logits, y)
        
        return {'loss':loss, 'val_acc':val_acc}
    
    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()
        avg_val_acc = torch.tensor([x['val_acc'] for x in val_step_outputs]).mean()
        
        metrics = {"val_loss":avg_val_loss, "val_acc":avg_val_acc}
        
        self.log_dict(metrics, prog_bar=True)
    
    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        acc = self.accuracy(logits, y)
        
        metrics = {'test_loss':loss, 'test_acc':acc}

        if stage:
            self.log_dict(metrics, prog_bar=True)
        
        return metrics
    
    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, 'test')

class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        return bar

# %%
if __name__ == '__main__':
    SIGNAL_DATAPATH = '../../data/RRI'
    MASTER_TABLE_DATAPATH = '../../data/master_table.csv'
    
    pl.seed_everything(1004)

    df_orig = pd.read_csv(MASTER_TABLE_DATAPATH)
    
    df_metric = pd.DataFrame()

    subject = list(set(df_orig['subject']))
    
    rkf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1004)
    
    for cv_num, (train_id, test_id) in enumerate(rkf.split(subject)):
        
        train_id, valid_id = train_test_split(train_id, test_size=1/9, random_state=1004)

        train_data = df_orig.query("subject.isin(@train_id) & session == [1, 2]", engine='python').reset_index(drop=True)
        valid_data = df_orig.query("subject.isin(@valid_id) & session == [1, 2]", engine='python').reset_index(drop=True)
        test_data = df_orig.query("subject.isin(@test_id) & session == [1, 2]", engine='python').reset_index(drop=True)
        
        train_dataset = CustomDataset(data_table=train_data, data_dir=SIGNAL_DATAPATH)
        valid_dataset = CustomDataset(data_table=valid_data, data_dir=SIGNAL_DATAPATH)
        test_dataset = CustomDataset(data_table=test_data, data_dir=SIGNAL_DATAPATH)

        trainset_loader = DataLoader(train_dataset, batch_size=130, shuffle=True, collate_fn=padd_seq, num_workers=4)
        validset_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, collate_fn=padd_seq, num_workers=4)
        testset_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=padd_seq, num_workers=4)
        
        bar = LitProgressBar()

        model = Stress_Classification()

        logger = TensorBoardLogger("tb_logs", name="cv", version=cv_num)
        
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                      dirpath='check_point/cv'+str(cv_num), 
                                      filename="residual_cnn_{epoch:03d}_{val_loss:.2f}", 
                                      save_top_k=3, 
                                      mode='min')

        trainer = pl.Trainer(logger=logger,
                             max_epochs=300,
                             accelerator='gpu', 
                             devices=[0], 
                             gradient_clip_val=0.3, 
                             log_every_n_steps=1, 
                             accumulate_grad_batches=1,
                             callbacks=[bar, checkpoint_callback])
        
        trainer.fit(model, 
                    train_dataloaders = trainset_loader, 
                    val_dataloaders = validset_loader)
        ## Retrieve the best model
        checkpoint_callback.best_model_path
        
        test_metric = trainer.test(test_dataloaders=testset_loader)
        
        df_metric = pd.concat((df_metric, pd.DataFrame([test_metric[0]])), axis=0)
        
        df_metric.reset_index(drop=True).to_csv("./cv_results.csv", index=False)