# %%
from turtle import end_fill
import pandas as pd

# %% Load data
master = pd.read_excel("../../data/Symptom_table.xlsx")
rri_table = pd.read_csv("../../data/master_table.csv")

outcome = pd.read_excel("../../data/depression_outcome.xlsx")

# %%
master = pd.merge(master, rri_table, how='left', on='subject')
master = pd.merge(master, outcome, how='left', on='subject')
master = master.sort_values(['subject', 'visit', 'session'])

# %%
master.to_csv("../../data/hrv_master_table.csv", index=False, encoding='utf-8-sig')
# %%

### Depression Detection Datatype
master_dp = pd.pivot(data=master, 
                     index=['subject', 'gender', 'visit', 'symptom', 'label'], 
                     values=['file_nm'], 
                     columns=['session'])

master_dp.columns = ['_'.join(['session', str(y)]) for x, y in master_dp.columns]

master_dp = master_dp.reset_index()
master_dp = master_dp.drop(columns=['session_nan'])
# %%
master_dp = master_dp.assign(count_nn = lambda x: x.loc[:, 'session_1.0': 'session_6.0'].notnull().sum(axis=1))

master_dp.to_csv("../../data/dep_master_table.csv", index=False, encoding='utf-8')
# %%
master_dp.query("label.notnull() & (count_nn == 6)", engine='python')