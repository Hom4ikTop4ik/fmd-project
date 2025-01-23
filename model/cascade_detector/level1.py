import torch.nn as nn

class Level1Detector(nn.Module):
    def __init__(self, device):
        super(Level1Detector, self).__init__()
        self.adpool = nn.AdaptiveAvgPool2d((128, 128)).to(device) 
        self.conv1 = nn.Conv2d(3, 6, 3, padding = 1).to(device)
        self.pool = nn.MaxPool2d(2, 2).to(device)
        self.conv2 = nn.Conv2d(6, 9, 3, padding = 1).to(device)
        self.conv3 = nn.Conv2d(9, 20, 3, padding = 1).to(device)
        self.conv4 = nn.Conv2d(20, 64, 3, padding = 1).to(device)
        self.fcsize = fcsize = 256
        self.fc1 = nn.Linear(64*16*16, fcsize).to(device)
        self.fc2 = nn.Linear(fcsize, fcsize).to(device)
        self.fc3 = nn.Linear(fcsize, fcsize).to(device)
        self.fc4 = nn.Linear(fcsize, 2 * 2).to(device)
        self.act = nn.ReLU().to(device)
    
    def forward(self, x):
        # Input: [batch, 3, H, W]
        x = self.adpool(x)
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = self.act(self.conv3(x))  
        x = self.pool(self.act(self.conv4(x)))
        x = x.view(-1, 64*16*16)       
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.fc4(x)
        x = x.view(-1, 2, 2)
        return x
