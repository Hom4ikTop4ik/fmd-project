import torch.nn as nn

class Level2Detector(nn.Module):
    def __init__(self, device):
        super(Level2Detector, self).__init__()
        self.adpool = nn.AdaptiveAvgPool2d((128, 128)).to(device) 
        self.pool = nn.MaxPool2d(2, 2).to(device)
        self.conv1 = nn.Conv2d(21, 8, 3, padding = 1).to(device)
        self.conv2 = nn.Conv2d(8, 16, 3, padding = 1).to(device)
        self.conv3 = nn.Conv2d(16, 32, 3, padding = 1).to(device)
        self.conv4 = nn.Conv2d(32, 32, 3, padding = 1).to(device)
        self.conv5 = nn.Conv2d(32, 64, 3, padding = 1).to(device)
        self.conv6 = nn.Conv2d(64, 128, 3, padding = 1).to(device)
        self.fcsize = fcsize = 256
        self.fc1 = nn.Linear(128*2*2, fcsize).to(device)
        self.fc2 = nn.Linear(fcsize, 2).to(device)
        self.act = nn.LeakyReLU(0.01).to(device)
    
    def forward(self, x):
        # Input: [batch, 3, H, W]
        x = self.adpool(x)
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = self.act(self.conv3(x))  
        x = self.pool(self.act(self.conv4(x)))
        x = self.pool(self.act(self.conv5(x)))
        x = self.pool(self.act(self.conv6(x)))
        x = self.pool(self.act(self.conv7(x)))
        x = x.view(-1, 128*2*2)       
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, 1, 2)
        return x
