# %% 
import torch
import torch.nn as nn
import pytorch_lightning as pl

# %%
class CNN_Res_Block(pl.LightningModule):
    def __init__(self, input_channels):
        super().__init__()
        
        self.cnn_block = nn.Sequential(nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=15, padding=7, stride=1),
                                       nn.BatchNorm1d(num_features=64),
                                       nn.ReLU(),
                                       nn.Dropout(),
                                       nn.Conv1d(in_channels=64, out_channels=64, kernel_size=15, padding=7, stride=1)
                                       )
        
        self.pool_layer = nn.AvgPool1d(kernel_size=1, stride=1)
        
    def forward(self, x):
        y_out = self.cnn_block(x) + self.pool_layer(x)
        
        return y_out
    
# %%
class Residual_Block(pl.LightningModule):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        
        self.conv_1_1 = nn.Conv1d(kernel_size=1, stride=1, in_channels=input_channels, out_channels=output_channels)
        
        self.conv_layers = nn.Sequential(nn.BatchNorm1d(num_features=input_channels), 
                                         nn.ReLU(), 
                                         nn.Dropout(),
                                         nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=15, stride=1, padding=7), 
                                         nn.BatchNorm1d(num_features=output_channels), 
                                         nn.ReLU(), 
                                         nn.Dropout(),
                                         nn.Conv1d(in_channels=output_channels, out_channels=output_channels, kernel_size=15, stride=1, padding=7))
        
    def forward(self, x):
        y_out = self.conv_layers(x) + self.conv_1_1(x)
        
        return y_out

# %%
class DNN_Model(pl.LightningModule):
    def __init__(self, output_class):
        super().__init__()
        
        self.cnn_block = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=64, kernel_size=15, stride=1, padding=7),
                                       nn.BatchNorm1d(num_features=64), 
                                       nn.ReLU())
        
        self.cnn_res_block = CNN_Res_Block(input_channels=64)
        
        self.res_block_1 = nn.ModuleList([Residual_Block(input_channels=64, output_channels=64) for i in range(4)])
        self.res_block_1_2 = Residual_Block(input_channels=64, output_channels=64*2)
        self.res_block_2 = nn.ModuleList([Residual_Block(input_channels=64*2, output_channels=64*2) for i in range(3)])
        self.res_block_2_3 = Residual_Block(input_channels=64*2, output_channels=64*3)
        self.res_block_3 = nn.ModuleList([Residual_Block(input_channels=64*3, output_channels=64*3) for i in range(3)])
        self.res_block_3_4 = Residual_Block(input_channels=64*3, output_channels=64*4)
        self.res_block_4 = nn.ModuleList([Residual_Block(input_channels=64*4, output_channels=64*4) for i in range(3)])
        
        self.linear = nn.Sequential(nn.BatchNorm1d(num_features=1200),
                                    nn.ReLU(),
                                    nn.Linear(1200, 1000),
                                    nn.BatchNorm1d(num_features=1000),
                                    nn.ReLU(),
                                    nn.Linear(1000, 1000),
                                    nn.BatchNorm1d(num_features=1000),
                                    nn.ReLU(),
                                    nn.Linear(1000, output_class))
        self.conv_1_1 = nn.Conv1d(in_channels=64*4, out_channels=1, kernel_size=1)
    
    def forward(self, x):
        y_cnn_block = self.cnn_block(x)
        y_cnn_res_block = self.cnn_res_block(y_cnn_block)
        for i, j in enumerate(self.res_block_1):
            if i == 0:
                y_h = y_cnn_res_block
            y_h = self.res_block_1[i](y_h)
        
        y_h = self.res_block_1_2(y_h)
        
        for i, j in enumerate(self.res_block_2):
            y_h = self.res_block_2[i](y_h)
        
        y_h = self.res_block_2_3(y_h)
        
        for i, j in enumerate(self.res_block_3):
            y_h = self.res_block_3[i](y_h)
        
        y_h = self.res_block_3_4(y_h)
        
        for i, j in enumerate(self.res_block_4):
            y_h = self.res_block_4[i](y_h)
            
        y_projected = nn.Flatten()(self.conv_1_1(y_h))
        
        y_out = self.linear(y_projected)
        
        
        return y_out
        
# %%
if __name__ == '__main__':
    test = torch.rand((2, 1*1200))
    
    cnn_layer = DNN_Model(output_class=2)
    tmp = cnn_layer.forward(test.view(2, 1, 1200))
    print(tmp.shape)
# %%
