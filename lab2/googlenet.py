import torch
from torch import nn


class InceptionBlock(nn.Module):
    def __init__(
            self, 
            channels: int, 
            c1x1: int,
            c3x3_reduced: int,
            c3x3: int,
            c5x5_reduced: int,
            c5x5: int,
            pool_proj: int
    ) -> None:
        
        super().__init__()

        self.left_branch = conv_block(
            in_channels=channels,
            out_channels=c1x1,
            kernel_size=1,
            padding=0,
        )

        self.middle_branch_1 = nn.Sequential(
            conv_block(
                in_channels=channels,
                out_channels=c3x3_reduced,
                kernel_size=1,
                padding=0,
            ),
            conv_block(
                in_channels=c3x3_reduced,
                out_channels=c3x3,
                kernel_size=3,
                padding=1,
            )
        )

        self.middle_branch_2 = nn.Sequential(
            conv_block(
                in_channels=channels,
                out_channels=c5x5_reduced,
                kernel_size=1,
                padding=0,
            ),
            conv_block(
                in_channels=c5x5_reduced,
                out_channels=c5x5,
                kernel_size=5,
                padding=2,
            )
        )    

        self.right_branch = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=3,
                stride=1,
                padding=1,
                ceil_mode=True,
            ),
            conv_block(
                in_channels=channels,
                out_channels=pool_proj,
                kernel_size=1,
                padding=0
            ),
        )
        

    def forward(self, x: torch.Tensor):
        left_features = self.left_branch(x)
        middle_1_features = self.middle_branch_1(x)
        middle_2_features = self.middle_branch_2(x)
        right_features = self.right_branch(x)

        out = torch.cat([left_features, middle_1_features, middle_2_features, right_features], dim=1)

        return out


def conv_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class GoogLeNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.conv_1 = conv_block(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3
        )
    
        self.max_pool_1 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            ceil_mode=True,
        )

        self.conv_2 = conv_block(
            in_channels=64,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.conv_3 = conv_block(
            in_channels=64,
            out_channels=192,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.max_pool_2 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            ceil_mode=True,
        )

        self.inception_3a = InceptionBlock(
            channels=192,
            c1x1=64,
            c3x3_reduced=96,
            c3x3=128,
            c5x5_reduced=16,
            c5x5=32,
            pool_proj=32,
        )

        self.inception_3b = InceptionBlock(
            channels=256,
            c1x1=128,
            c3x3_reduced=128,
            c3x3=192,
            c5x5_reduced=32,
            c5x5=96,
            pool_proj=64,
        )

        self.max_pool_3 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            ceil_mode=True
        )
        
        self.inception_4a = InceptionBlock(
            channels=480,
            c1x1=192,
            c3x3_reduced=96,
            c3x3=208,
            c5x5_reduced=16,
            c5x5=48,
            pool_proj=64,
        )

        self.inception_4b = InceptionBlock(
            channels=512,
            c1x1=160,
            c3x3_reduced=112,
            c3x3=224,
            c5x5_reduced=24,
            c5x5=64,
            pool_proj=64,
        )

        self.inception_4c = InceptionBlock(
            channels=512,
            c1x1=128,
            c3x3_reduced=128,
            c3x3=256,
            c5x5_reduced=24,
            c5x5=64,
            pool_proj=64,
        )

        self.inception_4d = InceptionBlock(
            channels=512,
            c1x1=112,
            c3x3_reduced=144,
            c3x3=288,
            c5x5_reduced=32,
            c5x5=64,
            pool_proj=64,
        )

        self.inception_4e = InceptionBlock(
            channels=528,
            c1x1=256,
            c3x3_reduced=160,
            c3x3=320,
            c5x5_reduced=32,
            c5x5=128,
            pool_proj=128,
        )

        self.max_pool_4 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            ceil_mode=True
        )


        self.inception_5a = InceptionBlock(
            channels=832,
            c1x1=256,
            c3x3_reduced=160,
            c3x3=320,
            c5x5_reduced=32,
            c5x5=128,
            pool_proj=128,
        )

        self.inception_5b = InceptionBlock(
            channels=832,
            c1x1=384,
            c3x3_reduced=192,
            c3x3=384,
            c5x5_reduced=48,
            c5x5=128,
            pool_proj=128,
        )

        self.avg_pool = nn.AvgPool2d(
            kernel_size=7,
            stride=1
        )


        self.drop_out = nn.Dropout(p=0.4)

        self.fc = nn.Linear(1024, num_classes)        

    def forward(self, x: torch.Tensor):

        out = self.conv_1(x)
        out = self.max_pool_1(out)
        out = self.conv_2(out)
        out = self.conv_3(out)

        out = self.max_pool_2(out)

        out = self.inception_3a(out)
        out = self.inception_3b(out)

        out = self.max_pool_3(out)

        out = self.inception_4a(out)
        out = self.inception_4b(out)
        out = self.inception_4c(out)
        out = self.inception_4d(out)
        out = self.inception_4e(out)

        out = self.max_pool_4(out)

        out = self.inception_5a(out)
        out = self.inception_5b(out)

        out = self.avg_pool(out)
        out = self.drop_out(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        
        return out

