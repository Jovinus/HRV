# %%
import pandas as pd

# %%
master = pd.read_excel("../data/Symptom_table.xlsx")
rri_table = pd.read_csv("../data/master_table.csv")

# %%
master = pd.merge(master, rri_table, how='left', on='subject')

master.to_csv("../data/hrv_master_table.csv", index=False, encoding='utf-8-sig')