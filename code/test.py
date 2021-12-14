# %%
import scipy.io as sio
import numpy as np
import pandas as pd
pd.set_option("display.max_columns",None)
# %%
test = sio.loadmat("../data/0Pooled_Data/E003_1_ECG_Session1_hrv.mat", uint16_codec='latin1')
# %%

## RRi
test['Res']['HRV'][0][0]['Data'][0][0]['RRi'][0][0][0][0]
# %%
