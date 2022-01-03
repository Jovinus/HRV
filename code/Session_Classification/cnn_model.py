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
                                       nn.MaxPool1d(kernel_size=4, stride=1, padding=1))
        
    def forward(self, x):
        y = torch.relu(self.cnn_block(x))
        return y

# %%
class CNN_FC_layer(pl.LightningModule):
    def __init__(self, output_class):
        super().__init__()
        
        self.cnn_blocks = nn.Sequential(CNN_Block(input_channels=1, output_channels=200),
                                        nn.BatchNorm1d(num_features=200),
                                        nn.ReLU(),
                                        CNN_Block(input_channels=200, output_channels=400),
                                        nn.BatchNorm1d(num_features=400),
                                        nn.ReLU(),
                                        CNN_Block(input_channels=400, output_channels=300),
                                        nn.BatchNorm1d(num_features=300),
                                        nn.ReLU(),
                                        # CNN_Block(input_channels=200, output_channels=150),
                                        # nn.ReLU(),
                                        CNN_Block(input_channels=300, output_channels=50),
                                        nn.BatchNorm1d(num_features=50),
                                        nn.ReLU(),
                                        nn.Flatten())
        
        self.linear = nn.Sequential(nn.Linear(11400, 1000),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(num_features=1000),
                                    nn.Linear(1000, 500),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(num_features=500),
                                    nn.Linear(500, output_class))
        
    
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
