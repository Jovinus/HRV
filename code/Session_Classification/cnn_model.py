# %% 
import torch
import torch.nn as nn
import pytorch_lightning as pl

# %%
class CNN_Block(pl.LightningModule):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        
        self.cnn_block = nn.Sequential(nn.Conv1d(in_channels=input_channels, out_channels=300, kernel_size=16, padding=8),
                                       nn.BatchNorm1d(num_features=300),
                                       nn.ReLU(), 
                                       nn.Conv1d(in_channels=300, out_channels=output_channels, kernel_size=16, padding=8),
                                       nn.BatchNorm1d(num_features=output_channels),
                                       nn.ReLU())
        
    def forward(self, x):
        y = torch.relu(self.cnn_block(x))
        return y

# %%
class CNN_FC_layer(pl.LightningModule):
    def __init__(self, output_class):
        super().__init__()
        
        self.cnn_blocks = nn.Sequential(CNN_Block(input_channels=1, output_channels=200),
                                        CNN_Block(input_channels=200, output_channels=400),
                                        nn.AvgPool1d(kernel_size=14, stride=1, padding=7),
                                        CNN_Block(input_channels=400, output_channels=500),
                                        CNN_Block(input_channels=500, output_channels=400),
                                        nn.AvgPool1d(kernel_size=14, stride=1, padding=7),
                                        CNN_Block(input_channels=400, output_channels=300),
                                        CNN_Block(input_channels=300, output_channels=200),
                                        nn.AvgPool1d(kernel_size=14, stride=1, padding=7),
                                        CNN_Block(input_channels=200, output_channels=100),
                                        CNN_Block(input_channels=100, output_channels=50),
                                        nn.AvgPool1d(kernel_size=14, stride=1, padding=7),
                                        nn.AvgPool2d(kernel_size=14),
                                        nn.Flatten())
        
        self.linear = nn.Sequential(nn.Linear(261, 100),
                                    nn.BatchNorm1d(num_features=100),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.Linear(100, 100),
                                    nn.BatchNorm1d(num_features=100),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.Linear(100, output_class))
        
    
    def forward(self, x):
        y_cnn_block = self.cnn_blocks(x)
        y_output = self.linear(y_cnn_block)

        return y_output
        
# %%

if __name__ == '__main__':
    test = torch.rand((2, 1200))
    cnn_layer = CNN_Block(input_channels=1, output_channels=10)
    tmp = cnn_layer.forward(test.view(2, 1, 1200))
    print(tmp)
    print(tmp.shape)
    
    cnn_layer = CNN_FC_layer(output_class=2)
    tmp = cnn_layer.forward(test.view(2, 1, 1200))
    print(tmp)
    print(tmp.shape)
# %%
