# %% 
import torch
import torch.nn as nn
import pytorch_lightning as pl
    
# %%
class Residual_Block(pl.LightningModule):
    def __init__(self, input_channels, output_channels, hidden_channels, stride=1):
        super().__init__()
        
        self.conv_1_1 = nn.Conv1d(kernel_size=1, stride=stride, in_channels=input_channels, out_channels=output_channels)
        
        self.conv_layers = nn.Sequential(nn.Conv1d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=15, stride=stride, padding=7), 
                                         nn.BatchNorm1d(num_features=hidden_channels), 
                                         nn.ReLU(),
                                         nn.Conv1d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=15, stride=1, padding=7), 
                                         nn.BatchNorm1d(num_features=output_channels), 
                                         nn.ReLU())
        
    def forward(self, x):
        y_out = self.conv_layers(x) + self.conv_1_1(x)
        
        return y_out

# %%
class Residual_CNN_Model(pl.LightningModule):
    def __init__(self, output_class):
        super().__init__()
        
        self.cnn_block = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=64, kernel_size=15, stride=2, padding=7),
                                       nn.BatchNorm1d(num_features=64), 
                                       nn.ReLU(),
                                       nn.MaxPool1d(kernel_size=5, stride=2))
        
        self.res_block_1 = nn.ModuleList([Residual_Block(input_channels=64*1, output_channels=64*1, hidden_channels=64*1, stride=1) for i in range(4)])
        self.res_block_2 = nn.ModuleList([Residual_Block(input_channels=64*1, output_channels=64*2, hidden_channels=64*2, stride=2)] + [Residual_Block(input_channels=64*2, output_channels=64*2, hidden_channels=64*2, stride=1) for i in range(3)])
        self.res_block_3 = nn.ModuleList([Residual_Block(input_channels=64*2, output_channels=64*3, hidden_channels=64*3, stride=2)] + [Residual_Block(input_channels=64*3, output_channels=64*3, hidden_channels=64*3, stride=1) for i in range(5)])
        self.res_block_4 = nn.ModuleList([Residual_Block(input_channels=64*3, output_channels=64*4, hidden_channels=64*4, stride=2)] + [Residual_Block(input_channels=64*4, output_channels=64*4, hidden_channels=64*4, stride=1) for i in range(2)])
        
        self.linear = nn.Sequential(nn.Linear(256, 100),
                                    nn.BatchNorm1d(num_features=100),
                                    nn.ReLU(),
                                    nn.Linear(100, 100),
                                    nn.BatchNorm1d(num_features=100),
                                    nn.ReLU(),
                                    nn.Linear(100, output_class))
        
        # self.conv_1_1 = nn.Conv1d(in_channels=64*4, out_channels=1, kernel_size=1)
        self.global_pool = nn.AvgPool1d(kernel_size=38)
    
    def forward(self, x):
        y_cnn_block = self.cnn_block(x)
        
        for i, j in enumerate(self.res_block_1):
            if i == 0:
                y_h = y_cnn_block
            y_h = self.res_block_1[i](y_h)
        
        for i, j in enumerate(self.res_block_2):
            y_h = self.res_block_2[i](y_h)
        
        for i, j in enumerate(self.res_block_3):
            y_h = self.res_block_3[i](y_h)
        
        for i, j in enumerate(self.res_block_4):
            y_h = self.res_block_4[i](y_h)
        
        y_pooled = self.global_pool(y_h)
        
        y_projected = nn.Flatten()(y_pooled)
        
        y_out = self.linear(y_projected)
        
        
        return y_out
        
# %%
if __name__ == '__main__':
    test = torch.rand((2, 1*1200))
    
    cnn_layer = Residual_CNN_Model(output_class=2)
    tmp = cnn_layer.forward(test.view(2, 1, 1200))
    print(tmp.shape)
# %%
