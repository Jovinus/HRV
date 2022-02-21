# %%
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import json
import os
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
# %%

class CustomDataset(Dataset):
    def __init__(self, data_table, data_dir) -> None:
        super().__init__()
        
        self.master = data_table
        self.data_dir = data_dir
        
    def __len__(self):
        return len(self.master)
    
    def __getitem__(self, idx):
        
        data_path = os.path.join(self.data_dir, self.master.loc[idx, 'session_1.0'])
        rri_1 = read_json_to_tensor(datapath=data_path)
        d_rri_1 = torch.diff(rri_1)
        
        data_path = os.path.join(self.data_dir, self.master.loc[idx, 'session_2.0'])
        rri_2 = read_json_to_tensor(datapath=data_path)
        d_rri_2 = torch.diff(rri_2)
        
        data_path = os.path.join(self.data_dir, self.master.loc[idx, 'session_3.0'])
        rri_3 = read_json_to_tensor(datapath=data_path)
        d_rri_3 = torch.diff(rri_3)
        
        data_path = os.path.join(self.data_dir, self.master.loc[idx, 'session_4.0'])
        rri_4 = read_json_to_tensor(datapath=data_path)
        d_rri_4 = torch.diff(rri_4)
        
        data_path = os.path.join(self.data_dir, self.master.loc[idx, 'session_5.0'])
        rri_5 = read_json_to_tensor(datapath=data_path)
        d_rri_5 = torch.diff(rri_5)
        
        rri = torch.cat((rri_1, rri_2, rri_3, rri_4, rri_5), dim=0)
        rri_mean, rri_std = torch.mean(rri), torch.std(rri)
        
        d_rri = torch.cat((d_rri_1, d_rri_2, d_rri_3, d_rri_4, d_rri_5), dim=0)
        d_rri_mean, d_rri_std = torch.mean(d_rri), torch.std(d_rri)
        
        ## Normalizing
        rri_1 = (rri_1 - rri_mean) / rri_std
        rri_2 = (rri_2 - rri_mean) / rri_std
        rri_3 = (rri_3 - rri_mean) / rri_std
        rri_4 = (rri_4 - rri_mean) / rri_std
        rri_5 = (rri_5 - rri_mean) / rri_std
        
        d_rri_1 = (d_rri_1 - d_rri_mean) / d_rri_std
        d_rri_2 = (d_rri_2 - d_rri_mean) / d_rri_std
        d_rri_3 = (d_rri_3 - d_rri_mean) / d_rri_std
        d_rri_4 = (d_rri_4 - d_rri_mean) / d_rri_std
        d_rri_5 = (d_rri_5 - d_rri_mean) / d_rri_std
        
        label = self.master.loc[idx, 'label']
        
        return  d_rri_1, d_rri_2, d_rri_3, d_rri_4, d_rri_5, label

# %%
def read_json_to_tensor(datapath):
    with open(datapath) as json_file:
            rri_json = json.load(json_file)
            
    rri = torch.from_numpy(np.array(rri_json['RRI'])).type(torch.float32)
    return rri

# %%
def padd_seq(batch):
    (x_1, x_2, x_3, x_4, x_5, y) = zip(*batch)
    y = torch.LongTensor(y)
    x_pad_1 = pad_sequence(x_1, batch_first=True, padding_value=0.0)
    x_pad_1 = pad(x_pad_1.view(x_pad_1.shape[0], 1, -1), (0, 1200 - x_pad_1.shape[1]), "constant", 0)
    
    x_pad_2 = pad_sequence(x_2, batch_first=True, padding_value=0.0)
    x_pad_2 = pad(x_pad_2.view(x_pad_2.shape[0], 1, -1), (0, 1200 - x_pad_2.shape[1]), "constant", 0)
    
    x_pad_3 = pad_sequence(x_3, batch_first=True, padding_value=0.0)
    x_pad_3 = pad(x_pad_3.view(x_pad_3.shape[0], 1, -1), (0, 1200 - x_pad_3.shape[1]), "constant", 0)
    
    x_pad_4 = pad_sequence(x_4, batch_first=True, padding_value=0.0)
    x_pad_4 = pad(x_pad_4.view(x_pad_4.shape[0], 1, -1), (0, 1200 - x_pad_4.shape[1]), "constant", 0)
    
    x_pad_5 = pad_sequence(x_5, batch_first=True, padding_value=0.0)
    x_pad_5 = pad(x_pad_5.view(x_pad_5.shape[0], 1, -1), (0, 1200 - x_pad_5.shape[1]), "constant", 0)
    
    x_pad = torch.cat((x_pad_1, x_pad_2, x_pad_3, x_pad_4, x_pad_5), 1)
    
    return x_pad, y

# %%

if __name__ == '__main__':
    DATAPATH = "../../data/RRI"
    table = pd.read_csv("../../data/dep_master_table.csv")
    table = table.query("visit == 1 & label.notnull() & count_nn == 6", engine='python').reset_index(drop=True)
# %%
    dataset = CustomDataset(data_table=table, data_dir=DATAPATH)

    for rri_1,  label in DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=padd_seq):
        print(rri_1.shape,  label.shape)

# %%
