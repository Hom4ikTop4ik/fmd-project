import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, device, insize, outsize):
        super(ConvBlock, self).__init__()
        self.act =  nn.ReLU().to(device)
        self.conv1 = nn.Conv2d(insize, insize, 3, padding = 1).to(device)
        self.conv2 = nn.Conv2d(insize, insize, 3, padding = 1).to(device)
        self.conv3 = nn.Conv2d(insize, outsize, 3, padding = 1).to(device)
        self.skipconv = nn.Conv2d(insize, outsize, 2, stride=2).to(device)
        self.pool = nn.MaxPool2d(2, 2).to(device)

    def forward(self, x):
        x1 = x.clone()
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        return self.pool(x) + self.skipconv(x1)

class Head(nn.Module):
    def __init__(self, device, insize, midsize, outsize):
        super(Head, self).__init__()
        self.insize = insize
        self.fc1 = nn.Linear(insize, midsize)
        self.fc2 = nn.Linear(midsize, outsize)
        self.act = nn.LeakyReLU(0.01).to(device)
    def forward(self, x):
        x = x.view(-1, self.insize)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


class MultyLayer(nn.Module):
    def __init__(self, device):
        super(MultyLayer, self).__init__()
        self.pulconv = nn.Conv2d(3, 3, 7, stride=2).to(device)

        self.cblock1 = ConvBlock(device, 3, 6) # now 128x128
        self.bn1 = nn.BatchNorm2d(6).to(device)
        self.cblock2 = ConvBlock(device, 6, 9) # now 64x64
        self.bn2 = nn.BatchNorm2d(9).to(device)
        self.cblock3 = ConvBlock(device, 9, 16) # now 32x32
        self.bn3 = nn.BatchNorm2d(16).to(device)
        self.cblock4 = ConvBlock(device, 16, 32) # now 16x16
        self.bn4 = nn.BatchNorm2d(32).to(device)
        self.cblock5 = ConvBlock(device, 32, 64) # now 8x8
        self.bn5 = nn.BatchNorm2d(64).to(device)
        self.cblock6 = ConvBlock(device, 64, 128) # now 4x4
        self.bn6 = nn.BatchNorm2d(128).to(device)
        self.cblock7 = ConvBlock(device, 128, 256) # now 1x1
        self.bn7 = nn.BatchNorm2d(256).to(device)

        self.head = Head(device, 256, 128, 40)

    def forward(self, x):
        # Input: [batch, 3, H, W]
        print(x.shape)
        x = self.pulconv(x)
        print(x.shape)
        x = self.bn1(self.cblock1(x))
        print(x.shape)
        x = self.bn2(self.cblock2(x))
        print(x.shape)
        x = self.bn3(self.cblock3(x))
        print(x.shape)
        x = self.bn4(self.cblock4(x))
        print(x.shape)
        x = self.bn5(self.cblock5(x))
        print(x.shape)
        x = self.bn6(self.cblock6(x))
        print(x.shape)
        x = self.bn7(self.cblock7(x))
        print(x.shape)
        x = self.head(x)
        print(x.shape)
        x = x.view(-1, 40)
        return x
