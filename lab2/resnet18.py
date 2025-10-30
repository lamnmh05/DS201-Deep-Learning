import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(out_channels,),
            
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(out_channels)
        )

        # Shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()


    
    def forward(self, x: torch.Tensor):
        block = self.block(x)
        shorcut = self.shortcut(x)

        out = block + shorcut
        out = F.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes = 21):
        super().__init__()

        self.num_classes = num_classes

        self.conv_1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn_1 = nn.BatchNorm2d(64)

        self.max_pool_1 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.layer_1 = self._make_layer(
            ResidualBlock, 
            in_channels=64,  
            out_channels=64,  
            num_blocks=2, 
            )
        
        self.layer_2 = self._make_layer(
            ResidualBlock, 
            in_channels=64,  
            out_channels=128, 
            num_blocks=2, 
            )
        
        self.layer_3 = self._make_layer(
            ResidualBlock, 
            in_channels=128, 
            out_channels=256, 
            num_blocks=2, 
            )
        
        self.layer_4 = self._make_layer(
            ResidualBlock, 
            in_channels=256, 
            out_channels=512, 
            num_blocks=2, 
            )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 

        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, in_channels, out_channels, num_blocks):
        layers = []

        layers.append(block(in_channels, out_channels, stride=1))

        for _ in range(1, num_blocks):
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
            layers.append(block(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)
            
    
    def forward(self, x: torch.Tensor):
        
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.max_pool_1(out)

        out = self.layer_1(out) # 64
        out = self.layer_2(out) # 128
        out = self.layer_3(out) # 256
        out = self.layer_4(out) # 512

        out = self.avgpool(out) # (B, 512, 1, 1)
        out = torch.flatten(out, 1) # (B, 512)
        out = self.fc(out)   
        
        return out
    

