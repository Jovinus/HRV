# %%
import scipy.io as sio
from tqdm import tqdm
import re
import json
from glob import glob
import pandas as pd
# %% Read Data List in Directory
data_list = glob('../data/0Pooled_Data/*mat')
data_list.sort()

# %% Read Matfile -> select rri data -> Save in JSON 
SAVEPATH = '../data/RRI/'
PATTERN = r"E(?P<subject>\d+)[_](?P<visit>\d+)[_]ECG[_]Session(?P<session>\d+)[_]hrv"

df_master = pd.DataFrame()

for file_nm in tqdm(data_list):
    info = re.search(PATTERN, file_nm)
    
    rri_mat = sio.loadmat(file_nm, uint16_codec='latin1')
    rri = {'subject':int(info.group('subject')),
           'visit':int(info.group('visit')),
           'session':int(info.group('session')),
           'RRI':rri_mat['Res']['HRV'][0][0]['Data'][0][0]['RRi'][0][0][0][0].tolist()}
    
    master = {'subject':[int(info.group('subject'))], 
              'visit':[int(info.group('visit'))], 
              'session':[int(info.group('session'))], 
              'file_nm':[file_nm[-27:-4] + '.json']}
    
    df_master = pd.concat((df_master, pd.DataFrame(master)), axis=0)
    
    with open(SAVEPATH + file_nm[-27:-4] + '.json', 'w') as fp:
        json.dump(rri, fp, indent=4)
        
df_master.to_csv("../data/master_table.csv", index=False, encoding='utf-8')