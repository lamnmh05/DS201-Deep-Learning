import torch
from torch import nn

class LeNet(nn.Module):
    def __init__ (self) -> None:
        super().__init__()
        
        self.model = nn.Sequential(
            # Convolution 1 (bs,1,28,28)->(bs,6,28,28)
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=(5,5),
                stride=1,
                padding=(2,2),
            ),
            
            nn.Sigmoid(),
            
            # Pooling 1 () (bs,6,28,28)->(bs,6,14,14)
            nn.AvgPool2d(
                stride=(2,2),
                kernel_size=(2,2),
                padding=(0,0),
            ),
            
            # Convolution 2 (bs,6,14,14)->(bs,16,10,10)
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=(5,5),
                padding=0,
            ),
            
            nn.Sigmoid(),
            
            # Pooling 2 (bs,16,10,10)->(bs,16,5,5)
            nn.AvgPool2d(
                stride=(2,2),
                kernel_size=(2,2),
                padding=(0,0),
            ),

            # Convolution 3 (bs,16,5,5)->(bs,120,1,1)
            nn.Conv2d(
                in_channels=16,
                out_channels=120,
                kernel_size=(5,5),
                padding=0,
            ),

            nn.Sigmoid(),

            # Flatten (bs,120,1,1)->(bs,120)
            nn.Flatten(
                start_dim=1,
                end_dim=-1,
            ),
            
            # Dense 1 (bs,120)->(bs,84)
            nn.Linear(
                in_features=120,
                out_features=84,
            ),
            
            nn.Sigmoid(),
            
            # Dense 2 (bs,84)->(bs,10)
            nn.Linear(
                in_features=84,
                out_features=10,
            ),

        )


    def forward(self, images: torch.Tensor) -> float:
        # images = images.unsqueeze(1) # (bs,28,28) -> (bs,1,28,28)
        return self.model(images)



if __name__ == '__main__':
    import torch
    test = torch.rand(1,2,2,2)
    print(test)
    print(test.shape)
    flat = nn.Flatten()
    conv = nn.Conv2d(2, 8, kernel_size=2, padding=0)
    print(flat(test))
    print(flat(test).shape)
    print(conv(test))
    print(conv(test).shape)
    print(flat(conv(test)))     
    print(flat(conv(test)).shape)

