# %%
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn as nn
import torch
from torchmetrics import Accuracy
from my_dataloader import *
from torch.utils.data import DataLoader
from residual_cnn_1d_2 import *
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RepeatedKFold
from custom_loss import *
from my_module import *
# %%
class Depression_Detection(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.loss = Cosine_Loss()
        self.softmax = nn.Softmax(dim=1)
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
        acc = self.accuracy(torch.argmax(logits, dim=1), y)
        
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        
        val_acc = self.accuracy(torch.argmax(logits, dim=1), y)
        metrics = {"val_loss":loss, "val_acc":val_acc}
        self.log_dict(metrics, prog_bar=True)
        
        return {'loss':loss, 'val_acc':val_acc}
    
    def evaluate(self, y_true, y_pred, stage=None):
        logits = y_pred
        loss = self.loss(logits, y_true)
        acc = self.accuracy(torch.argmax(logits, dim=1), y_true)
        
        logits = logits.cpu().numpy()[:, -1]
        y = y_true.cpu().numpy()
        specificity, sensitivity, ppv, npv, f1, accuracy, threshold_of_interest, roc_auc, pr_auc, _, _, _, _ = performances_hard_decision(y, logits, youden=True)
        
        metrics = {'test_loss':loss, 'test_acc':acc, 
                   'test_specificity':specificity, "test_sensitvity":sensitivity, 
                   "test_ppv":ppv, "test_npv":npv, 
                   "test_f1":f1, "test_acc_youden":accuracy, 
                   "test_threshold":threshold_of_interest, 
                   "test_auroc":roc_auc, "test_auprc":pr_auc}
        
        if stage:
            self.log_dict(metrics, prog_bar=True)
        
        return metrics
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        return self.evaluate(y_true=y, y_pred=logits, stage='test')
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        return logits, y, torch.argmax(logits, dim=1)

class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        return bar

# %%
if __name__ == '__main__':
    SIGNAL_DATAPATH = '../../data/RRI'
    MASTER_TABLE_DATAPATH = '../../data/dep_master_table.csv'
    
    pl.seed_everything(1004, workers=True)

    df_orig = pd.read_csv(MASTER_TABLE_DATAPATH)
    
    df_metric = pd.DataFrame()
    df_con_matrix = pd.DataFrame()
    df_fig_curve = pd.DataFrame()

    subject = df_orig.query("visit == 1 & label.notnull() & count_nn == 6", engine='python').reset_index(drop=True)
    
    rkf_outer = RepeatedKFold(n_splits=78, n_repeats=1, random_state=1004)
    rkf_inner = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1004)
    
    for cv_num_outer, (train_id, test_id) in enumerate(rkf_outer.split(subject, y=subject['label'])):
        
        for cv_num_inner, (train_id, valid_id) in enumerate(rkf_inner.split(subject.iloc[train_id], y=subject.iloc[train_id]['label'])):
        
            train_id, valid_id = train_test_split(subject.loc[train_id, 'subject'], test_size=1/9, random_state=1004, stratify=subject.loc[train_id, 'label'])
            test_id = subject.loc[test_id, 'subject']
            
            train_data = df_orig.query("subject.isin(@train_id) & visit == 1", engine='python').reset_index(drop=True)
            valid_data = df_orig.query("subject.isin(@valid_id) & visit == 1", engine='python').reset_index(drop=True)
            test_data = df_orig.query("subject.isin(@test_id) & visit == 1", engine='python').reset_index(drop=True)
            
            train_dataset = CustomDataset(data_table=train_data, data_dir=SIGNAL_DATAPATH)
            valid_dataset = CustomDataset(data_table=valid_data, data_dir=SIGNAL_DATAPATH)
            test_dataset = CustomDataset(data_table=test_data, data_dir=SIGNAL_DATAPATH)
            
            trainset_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=padd_seq, num_workers=4)
            validset_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=padd_seq, num_workers=4)
            testset_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=padd_seq, num_workers=1)
            
            bar = LitProgressBar()
            
            model = Depression_Detection()
            
            logger = TensorBoardLogger("tb_logs", name="dep_test_loocv_1", version=cv_num_outer)
            
            checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                        dirpath='check_point/dep_test_loocv_1_outer'+str(cv_num_outer)+'_inner_'+str(cv_num_inner), 
                                        filename="residual_cnn_{epoch:03d}_{val_loss:.2f}", 
                                        save_top_k=3, 
                                        mode='min')
            
            trainer = pl.Trainer(logger=logger,
                                max_epochs=400,
                                accelerator='gpu', 
                                devices=[0], 
                                gradient_clip_val=0.3, 
                                log_every_n_steps=1, 
                                accumulate_grad_batches=2,
                                callbacks=[bar, checkpoint_callback], 
                                deterministic=True)
            
            trainer.fit(model, 
                        train_dataloaders = trainset_loader, 
                        val_dataloaders = validset_loader)
            
            ## Retrieve the best model
            checkpoint_callback.best_model_path
        
            ## Log Prediction and Label
            results = trainer.predict(model, dataloaders=testset_loader)
            pred_proba =  torch.vstack([results[i][0] for i in range(len(results))]).cpu().numpy()[:, 1].tolist()
            labels = torch.hstack([results[i][1] for i in range(len(results))]).cpu().numpy().tolist()
            preds = torch.hstack([results[i][2] for i in range(len(results))]).cpu().numpy().tolist()
            pred_log = pd.DataFrame({'pred_proba':pred_proba, 'label':labels, 'pred':preds})
            pred_log['cv_num_outer'] = cv_num_outer
            pred_log['cv_num_inner'] = cv_num_inner
            
            df_con_matrix = pd.concat((df_con_matrix, pred_log), axis=0)
            
    df_con_matrix.reset_index(drop=True).to_csv("./dep_test_loocv_1_conf_pred_results_final.csv", index=False)
# %%
