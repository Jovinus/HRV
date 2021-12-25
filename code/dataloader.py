# %%
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import json
import os
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence
from torch.nn.functional import pad
# %%

class CustomDataset(Dataset):
    def __init__(self, data_dir) -> None:
        super().__init__()
        
        self.master = pd.read_csv("../data/master_table.csv").query('session == [1, 2, 3]', engine='python').reset_index(drop=True)
        self.data_dir = data_dir
        
    def __len__(self):
        return len(self.master)
    
    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.master.loc[idx, 'file_nm'])
        
        with open(data_path) as json_file:
            rri_json = json.load(json_file)
            
        rri = torch.from_numpy(np.array(rri_json['RRI']))
        label = self.master.loc[idx, 'session']
        
        return  rri, label

# %%
def padd_seq(batch):
    (x, y) = zip(*batch)
    y = torch.LongTensor(y)
    x_pad = pad_sequence(x, batch_first=True, padding_value=0.0)
    return x_pad, y

# %%

if __name__ == '__main__':
    dataset = CustomDataset("../data/RRI")
    for rri, label in DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=padd_seq):
        print(rri.shape, label.shape)
    
    ## pad to fixed length
    pad(rri.view(18, 1, -1), (0, 1500 - rri.shape[1]), "constant", 0)
# %%
