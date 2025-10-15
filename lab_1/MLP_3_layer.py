from torch import nn

class MLP3Layer(nn.Module):
    def __init__(self, img_size: tuple, num_labels: int) -> None:
        super().__init__()

        w, h = img_size
        input_size = w * h
        self.fc1 = nn.Linear(in_features= input_size, out_features=512)
        self.fc2 = nn.Linear(in_features= 512, out_features= 256)
        self.fc3 = nn.Linear(in_features= 256, out_features= num_labels)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        out = self.fc1(x)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.softmax(out)
        
        return out