# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import re
from my_module import *
from IPython.display import display
pd.set_option("display.max_columns", None)

# %% Confusion Matrix

df_conf = pd.read_csv("./stress_final_1_conf_pred_results_final.csv")
display(pd.crosstab(df_conf.pred, df_conf.label))

df_conf = pd.read_csv("./deep_stress_final_1_conf_pred_results_final.csv")
display(pd.crosstab(df_conf.pred, df_conf.label))

df_conf = pd.read_csv("./cnn_gru_stress_final_1_conf_pred_results_final.csv")
display(pd.crosstab(df_conf.pred, df_conf.label))
# %% AUROC Plot
df_fig_1d = pd.read_csv("./stress_final_1_fig_results_final.csv")
df_fig_1d['feature'] = "Model 1"
df_metrics_1d = pd.read_csv("./stress_final_1_results_final.csv")
df_metrics_1d['feature'] = "Model 1"


df_fig_deep = pd.read_csv("./deep_stress_final_1_fig_results_final.csv")
df_fig_deep['feature'] = "Model 2"
df_metrics_deep = pd.read_csv("./deep_stress_final_1_results_final.csv")
df_metrics_deep['feature'] = "Model 2"

df_fig_cnn_gru = pd.read_csv("./cnn_gru_stress_final_1_fig_results_final.csv")
df_fig_cnn_gru['feature'] = "Model 3"
df_metrics_cnn_gru = pd.read_csv("./cnn_gru_stress_final_1_results_final.csv")
df_metrics_cnn_gru['feature'] = "Model 3"

df_fig = pd.concat((df_fig_1d, df_fig_deep, df_fig_cnn_gru), axis=0).reset_index(drop=True)
df_metrics = pd.concat((df_metrics_1d, df_metrics_deep, df_metrics_cnn_gru), axis=0).reset_index(drop=True)

# %%
plot_roc_curve(x='fpr', y='tpr', data=df_fig_deep, mean=df_metrics_deep['test_auroc'].mean(), std=df_metrics_deep['test_auroc'].std())
plot_prc_curve(x='recall', y='precision', data=df_fig_deep, mean=df_metrics_deep['test_auprc'].mean(), std=df_metrics_deep['test_auprc'].std())
   
# %%
feature_order = ['Model 1', 'Model 2', 'Model 3']
plot_roc_curve_in_one(x='fpr', y='tpr', data=df_fig, metric=df_metrics, feature_order=feature_order)
plot_prc_curve_in_one(x='recall', y='precision', data=df_fig, metric=df_metrics, feature_order=feature_order)

# %%
