# %% 
import torch
import torch.nn as nn
import pytorch_lightning as pl

# %%
class CNN_Block(pl.LightningModule):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        
        self.cnn_block = nn.Sequential(nn.Conv1d(in_channels=input_channels, out_channels=100, kernel_size=124, padding=1), 
                                       nn.Conv1d(in_channels=100, out_channels=output_channels, kernel_size=124, padding=1),
                                       nn.AvgPool1d(kernel_size=100, stride=1, padding=1))
        
    def forward(self, x):
        y = torch.relu(self.cnn_block(x))
        return y

# %%
class CNN_FC_layer(pl.LightningModule):
    def __init__(self, output_class):
        super().__init__()
        
        self.cnn_blocks = nn.Sequential(CNN_Block(input_channels=1, output_channels=100),
                                        nn.ReLU(), 
                                        CNN_Block(input_channels=100, output_channels=200),
                                        nn.ReLU(),
                                        # CNN_Block(input_channels=200, output_channels=100),
                                        # nn.ReLU(),
                                        # CNN_Block(input_channels=200, output_channels=50),
                                        # nn.ReLU(),
                                        CNN_Block(input_channels=200, output_channels=10),
                                        nn.ReLU(),
                                        nn.Flatten())
        
        self.linear = nn.Sequential(nn.Linear(1830, 1000),
                                    nn.Dropout(p=0.5),
                                    nn.ReLU(),
                                    nn.Linear(1000, 100),
                                    nn.Dropout(p=0.5),
                                    nn.ReLU(),
                                    nn.Linear(100, output_class), 
                                    nn.Dropout(p=0.5),)
        
    
    def forward(self, x):
        y_cnn_block = self.cnn_blocks(x)
        y_output = self.linear(y_cnn_block)

        return y_output
        
# %%

if __name__ == '__main__':
    test = torch.rand((1, 1200))
    cnn_layer = CNN_Block(input_channels=1, output_channels=10)
    tmp = cnn_layer.forward(test.view(1, 1, 1200))
    print(tmp)
    print(tmp.shape)
    
    cnn_layer = CNN_FC_layer(output_class=2)
    tmp = cnn_layer.forward(test.view(1, 1, 1200))
    print(tmp)
    print(tmp.shape)
# %%
